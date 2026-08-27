use bevy::{
    math::USizeVec3,
    platform::collections::{HashMap, HashSet},
    prelude::*,
};
use ndshape::{ConstPow2Shape3usize, ConstShape as _};

const SIZE_LOG2: usize = 5;
const SIZE: usize = 1 << SIZE_LOG2;
const SIZE_3D: usize = SIZE * SIZE * SIZE;
const CELL_SIZE: f32 = 16.0;

type Field<T> = [T; SIZE_3D];
type Shape = ConstPow2Shape3usize<SIZE_LOG2, SIZE_LOG2, SIZE_LOG2>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Id {
    Rigid,
}

/// Sparse chunks containing dense 3D fields.
#[derive(Component, Default)]
struct Storage {
    map: HashMap<IVec3, usize>,
    inverse: Vec<IVec3>,
    id: Vec<Box<Field<Option<Id>>>>,
}

impl Storage {
    fn get(&self, pos: IVec3) -> Option<Id> {
        let (chunk, local) = self.index_info(pos);
        self.id[chunk?][local]
    }

    fn force_set(&mut self, pos: IVec3, id: Option<Id>) {
        self.load_for_pos(pos);
        let (chunk, local) = self.index_info(pos);
        self.id[chunk.expect("force_set always loads its chunk")][local] = id;
    }

    /// Creates the MVP object in the XY plane. The storage and projection remain fully 3D.
    fn non_empty_default() -> Self {
        let mut storage = Self::default();

        // Deliberately use only z = 0 for the initial scene. The third dimension is
        // still represented by every field and is used by the projection below.
        for y in -8i32..=8 {
            for x in -11i32..=11 {
                let in_disc = x * x + y * y <= 64;
                let in_bar = x.abs() <= 10 && y.abs() <= 2;
                if in_disc || in_bar {
                    storage.force_set(IVec3::new(x, y, 0), Some(Id::Rigid));
                }
            }
        }

        storage
    }

    fn iter_set(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.map.iter().flat_map(|(&chunk, &index)| {
            let ids = &*self.id[index];
            let global = chunk * SIZE as i32;
            ids.iter().enumerate().filter_map(move |(index, id)| {
                id.map(|_| {
                    let local = USizeVec3::from(Shape::delinearize(index)).as_ivec3();
                    global + local
                })
            })
        })
    }

    fn clear(&mut self) {
        self.map.clear();
        self.inverse.clear();
        self.id.clear();
    }

    fn load_for_pos(&mut self, pos: IVec3) {
        self.load(pos.div_euclid(IVec3::splat(SIZE as i32)));
    }

    /// Marks a chunk as editable and allocates its dense 3D field.
    fn load(&mut self, pos: IVec3) {
        self.map.entry(pos).or_insert_with(|| {
            self.id
                .push(vec![None; SIZE_3D].into_boxed_slice().try_into().unwrap());
            self.inverse.push(pos);
            self.id.len() - 1
        });
    }

    #[allow(dead_code)]
    fn unload(&mut self, pos: IVec3) {
        if let Some(index) = self.map.remove(&pos) {
            self.inverse.swap_remove(index);
            self.id.swap_remove(index);

            // Keep the sparse map valid when swap_remove moved another chunk.
            if let Some(&moved) = self.inverse.get(index) {
                self.map.insert(moved, index);
            }
        }
    }

    #[inline]
    fn index_info(&self, pos: IVec3) -> (Option<usize>, usize) {
        const CHUNK_SIZE: IVec3 = IVec3::splat(SIZE as i32);
        let chunk = pos.div_euclid(CHUNK_SIZE);
        let local = pos.rem_euclid(CHUNK_SIZE);
        let local = Shape::linearize(local.as_usizevec3().to_array());
        (self.map.get(&chunk).copied(), local)
    }
}

#[derive(Component)]
struct Target;

#[derive(Component)]
struct Sources;

#[derive(Component)]
struct SourceMotion {
    origin: Vec3,
    phase: f32,
}

#[derive(Component)]
struct RenderedVoxel {
    position: IVec3,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, (move_sources, project, render_target).chain())
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);

    commands.spawn((Storage::default(), Target));
    commands.spawn((
        Storage::non_empty_default(),
        Sources,
        Transform::from_translation(Vec3::ZERO),
        SourceMotion {
            origin: Vec3::ZERO,
            phase: 0.0,
        },
    ));
}

/// Moves and rotates the source continuously in the XY plane. Its z position and rotation
/// axis intentionally remain fixed for the MVP, while the transform and projection are still
/// 3D-capable.
fn move_sources(
    time: Res<Time>,
    mut sources: Query<(&mut Transform, &SourceMotion), With<Sources>>,
) {
    let elapsed = time.elapsed_secs();
    for (mut transform, motion) in &mut sources {
        transform.translation = motion.origin
            + Vec3::new(
                elapsed.sin() * 7.0,
                (elapsed * 0.73 + motion.phase).cos() * 4.0,
                0.0,
            );
        transform.rotation = Quat::from_rotation_z(elapsed * 0.35 + motion.phase);
    }
}

#[allow(clippy::type_complexity)]
fn project(
    mut target: Query<&mut Storage, (With<Target>, Without<Sources>)>,
    sources: Query<(&Transform, &Storage), (With<Sources>, Without<Target>)>,
) {
    let Ok(mut target) = target.single_mut() else {
        return;
    };
    target.clear();

    for (transform, storage) in sources {
        let mut graph = mcmf::GraphBuilder::<ProjectionNode>::new();
        let mut sinks = HashSet::new();

        // this should later use direct indexing
        let mut trgt_weight = HashMap::<IVec3, u8>::new();

        for point in storage.iter_set() {
            // Integer voxel coordinates identify cells, so transform their centers.
            // Keeping `point` here is important: it is the source storage coordinate
            // needed to recover the payload after the matching graph is solved.
            let cont =
                (transform.to_matrix() * (point.as_vec3() + Vec3::splat(0.5)).extend(1.0)).xyz();
            let discrete = cont.floor().as_ivec3();
            let source_node = ProjectionNode::Source(OrdIVec3(point));

            // Candidate cells are searched in all three dimensions. Rendering later
            // intentionally drops z, but z is never dropped from simulation/matching.
            for x in -2..=2 {
                for y in -2..=2 {
                    for z in -2..=2 {
                        let vec = IVec3::new(x, y, z);
                        let target_pos = discrete + vec;
                        let center = target_pos.as_vec3() + Vec3::splat(0.5);
                        let distance = (cont.distance_squared(center) * 10000.0) as i32;

                        let target_node = ProjectionNode::Target(OrdIVec3(target_pos));
                        graph.add_edge(
                            source_node,
                            target_node,
                            mcmf::Capacity(1),
                            mcmf::Cost(distance),
                        );
                        if sinks.insert(target_node) {
                            graph.add_edge(
                                target_node,
                                mcmf::Vertex::Sink,
                                mcmf::Capacity(1),
                                mcmf::Cost(0),
                            );
                        }
                        if vec == IVec3::ZERO {
                            *trgt_weight.entry(target_pos).or_default() += 2;
                        } else {
                            *trgt_weight.entry(target_pos).or_default() += 1;
                        }
                    }
                }
            }

            graph.add_edge(
                mcmf::Vertex::Source,
                source_node,
                mcmf::Capacity(1),
                mcmf::Cost(0),
            );
        }

        for (from, _, _, cost) in &mut graph.edge_list {
            if let mcmf::Vertex::Node(ProjectionNode::Target(OrdIVec3(target_pos))) = from {
                cost.0 *= (*trgt_weight.get(target_pos).unwrap()) as i32;
            }
        }

        let (_, paths) = graph.mcmf();
        for path in paths {
            let &mcmf::Vertex::Node(ProjectionNode::Source(OrdIVec3(from))) = path.vertices()[1]
            else {
                continue;
            };
            let &mcmf::Vertex::Node(ProjectionNode::Target(OrdIVec3(to))) = path.vertices()[2]
            else {
                continue;
            };

            target.force_set(to, storage.get(from));
        }
    }
}

fn render_target(
    mut commands: Commands,
    target: Query<&Storage, With<Target>>,
    mut rendered: Query<(Entity, &RenderedVoxel, &mut Sprite, &mut Transform)>,
) {
    let Ok(target) = target.single() else {
        return;
    };

    let mut remaining: HashSet<IVec3> = target.iter_set().collect();

    for (entity, voxel, mut sprite, mut transform) in &mut rendered {
        if remaining.remove(&voxel.position) {
            sprite.color = voxel_color(target.get(voxel.position));
            transform.translation = screen_position(voxel.position);
        } else {
            commands.entity(entity).despawn();
        }
    }

    for position in remaining {
        commands.spawn((
            Sprite::from_color(
                voxel_color(target.get(position)),
                Vec2::splat(CELL_SIZE - 1.0),
            ),
            RenderedVoxel { position },
            Transform::from_translation(screen_position(position)),
        ));
    }
}

/// The renderer is deliberately 2D: z affects neither screen position nor cell size.
fn screen_position(position: IVec3) -> Vec3 {
    Vec3::new(
        position.x as f32 * CELL_SIZE,
        position.y as f32 * CELL_SIZE,
        0.0,
    )
}

fn voxel_color(id: Option<Id>) -> Color {
    match id {
        Some(Id::Rigid) => Color::srgb(0.15, 0.75, 1.0),
        None => Color::NONE,
    }
}

/// Source and target cells need separate graph identities even when their coordinates match.
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
enum ProjectionNode {
    Source(OrdIVec3),
    Target(OrdIVec3),
}

impl PartialOrd for ProjectionNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProjectionNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::Source(left), Self::Source(right))
            | (Self::Target(left), Self::Target(right)) => left.cmp(right),
            (Self::Source(_), Self::Target(_)) => std::cmp::Ordering::Less,
            (Self::Target(_), Self::Source(_)) => std::cmp::Ordering::Greater,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct OrdIVec3(IVec3);

impl PartialOrd for OrdIVec3 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdIVec3 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .x
            .cmp(&other.0.x)
            .then(self.0.y.cmp(&other.0.y))
            .then(self.0.z.cmp(&other.0.z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_supports_3d_positions_and_negative_chunks() {
        let mut storage = Storage::default();
        let positions = [
            IVec3::new(0, 0, 0),
            IVec3::new(31, 31, 31),
            IVec3::new(32, 32, 32),
            IVec3::new(-1, -1, -1),
            IVec3::new(-32, -32, -32),
            IVec3::new(-33, -33, -33),
        ];

        for position in positions {
            storage.force_set(position, Some(Id::Rigid));
        }

        for position in positions {
            assert_eq!(storage.get(position), Some(Id::Rigid));
        }
        assert_eq!(storage.iter_set().count(), positions.len());
    }

    #[test]
    fn projection_graph_allows_an_exact_coordinate_match() {
        let position = IVec3::ZERO;
        let source = ProjectionNode::Source(OrdIVec3(position));
        let target = ProjectionNode::Target(OrdIVec3(position));
        let mut graph = mcmf::GraphBuilder::<ProjectionNode>::new();

        graph.add_edge(
            mcmf::Vertex::Source,
            source,
            mcmf::Capacity(1),
            mcmf::Cost(0),
        );
        graph.add_edge(source, target, mcmf::Capacity(1), mcmf::Cost(0));
        graph.add_edge(target, mcmf::Vertex::Sink, mcmf::Capacity(1), mcmf::Cost(0));

        let (_, paths) = graph.mcmf();
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn projection_preserves_a_moved_voxel() {
        let mut world = World::new();
        let target_entity = world.spawn((Storage::default(), Target)).id();

        let mut source_storage = Storage::default();
        source_storage.force_set(IVec3::ZERO, Some(Id::Rigid));
        world.spawn((source_storage, Sources, Transform::from_xyz(4.25, 0.0, 0.0)));

        let mut schedule = Schedule::default();
        schedule.add_systems(project);
        schedule.run(&mut world);

        let target = world.get::<Storage>(target_entity).unwrap();
        assert_eq!(target.iter_set().count(), 1);
        assert_eq!(target.get(IVec3::new(4, 0, 0)), Some(Id::Rigid));
    }

    #[test]
    fn unloading_keeps_remaining_chunk_indices_valid() {
        let mut storage = Storage::default();
        storage.force_set(IVec3::ZERO, Some(Id::Rigid));
        storage.force_set(IVec3::new(SIZE as i32, 0, 0), Some(Id::Rigid));
        storage.force_set(IVec3::new(0, SIZE as i32, 0), Some(Id::Rigid));

        storage.unload(IVec3::ZERO);

        assert_eq!(storage.get(IVec3::new(SIZE as i32, 0, 0)), Some(Id::Rigid));
        assert_eq!(storage.get(IVec3::new(0, SIZE as i32, 0)), Some(Id::Rigid));
    }
}
