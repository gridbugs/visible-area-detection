use direction::DirectionBitmap;
pub use grid_2d::GridEnumerate;
use grid_2d::{Coord, CoordIter, Grid, Size};
use rgb_int::Rgb24;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
pub use shadowcast::{vision_distance, VisionDistance};
use shadowcast::{Context as ShadowcastContext, InputGrid};
use std::marker::PhantomData;

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Rational {
    pub numerator: u32,
    pub denominator: u32,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Light<V: VisionDistance> {
    pub colour: Rgb24,
    pub vision_distance: V,
    pub diminish: Rational,
}

pub trait World {
    type VisionDistance: VisionDistance;
    fn size(&self) -> Size;
    fn get_opacity(&self, coord: Coord) -> u8;
    fn for_each_light_by_coord<F: FnMut(Coord, &Light<Self::VisionDistance>)>(&self, _: F) {}
}

struct Visibility<W: World>(PhantomData<W>);

impl<W: World> InputGrid for Visibility<W> {
    type Grid = W;
    type Opacity = u8;
    fn size(&self, world: &Self::Grid) -> Size {
        world.size()
    }
    fn get_opacity(&self, grid: &Self::Grid, coord: Coord) -> Self::Opacity {
        grid.get_opacity(coord)
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CellVisibility<T> {
    Never,
    Previous(T),
    Current {
        data: T,
        light_colour: Option<Rgb24>,
    },
}

impl<T> CellVisibility<T> {
    pub fn into_data(self) -> Option<T> {
        match self {
            Self::Never => None,
            Self::Previous(data) | Self::Current { data, .. } => Some(data),
        }
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
struct VisibilityCell<T: Default> {
    last_seen: u64,
    last_lit: u64,
    visible_directions: DirectionBitmap,
    light_colour: Rgb24,
    data: T,
}

impl<T: Default> Default for VisibilityCell<T> {
    fn default() -> Self {
        Self {
            last_seen: 0,
            last_lit: 0,
            visible_directions: DirectionBitmap::default(),
            light_colour: Rgb24::new_grey(0),
            data: Default::default(),
        }
    }
}

impl<T: Default> VisibilityCell<T> {
    fn visibility(&self, count: u64) -> CellVisibility<&T> {
        if self.last_seen == count {
            let light_colour = if self.last_lit == count {
                Some(self.light_colour)
            } else {
                None
            };
            CellVisibility::Current {
                data: &self.data,
                light_colour,
            }
        } else if self.last_seen == 0 {
            CellVisibility::Never
        } else {
            CellVisibility::Previous(&self.data)
        }
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct VisibilityGrid<T: Default = ()> {
    grid: Grid<VisibilityCell<T>>,
    count: u64,
    shadowcast_context: ShadowcastContext<u8>,
}

impl<T: Default> VisibilityGrid<T> {
    pub fn new(size: Size) -> Self {
        Self {
            grid: Grid::new_default(size),
            count: 0,
            shadowcast_context: Default::default(),
        }
    }

    fn apply_lights<W: World>(&mut self, world: &W) {
        let visibility: Visibility<W> = Visibility(PhantomData);
        world.for_each_light_by_coord(|light_coord, light| {
            self.shadowcast_context.for_each_visible(
                light_coord,
                &visibility,
                world,
                light.vision_distance,
                255,
                |cell_coord, visible_directions, visibility| {
                    let cell = self.grid.get_checked_mut(cell_coord);
                    if cell.last_seen == self.count
                        && !(visible_directions & cell.visible_directions).is_empty()
                    {
                        cell.last_lit = self.count;
                        let distance_squared = (light_coord - cell_coord).magnitude2();
                        let light_colour = light.colour.saturating_scalar_mul_div(
                            light.diminish.denominator,
                            distance_squared.max(1) * light.diminish.numerator,
                        );
                        cell.light_colour = cell
                            .light_colour
                            .saturating_add(light_colour.normalised_scalar_mul(visibility));
                    }
                },
            );
        });
    }

    pub fn update<W: World, V: VisionDistance>(
        &mut self,
        ambient_light_colour: Rgb24,
        world: &W,
        vision_distance: V,
        eye: Coord,
    ) {
        self.update_custom(ambient_light_colour, world, vision_distance, eye, |_, _| {});
    }

    pub fn update_custom<W: World, V: VisionDistance, F: FnMut(&mut T, Coord)>(
        &mut self,
        ambient_light_colour: Rgb24,
        world: &W,
        vision_distance: V,
        eye: Coord,
        mut f: F,
    ) {
        self.count += 1;
        let visibility: Visibility<W> = Visibility(PhantomData);
        self.shadowcast_context.for_each_visible(
            eye,
            &visibility,
            world,
            vision_distance,
            255,
            |coord, visible_directions, _visibility| {
                let cell = self.grid.get_checked_mut(coord);
                cell.last_seen = self.count;
                cell.light_colour = ambient_light_colour;
                cell.visible_directions = visible_directions;
                f(&mut cell.data, coord);
            },
        );
        self.apply_lights(world);
    }

    pub fn update_omniscient<W: World>(&mut self, ambient_light_colour: Rgb24, world: &W) {
        self.update_omniscient_custom(ambient_light_colour, world, |_, _| {})
    }

    pub fn update_omniscient_custom<W: World, F: FnMut(&mut T, Coord)>(
        &mut self,
        ambient_light_colour: Rgb24,
        world: &W,
        mut f: F,
    ) {
        self.count += 1;
        for coord in CoordIter::new(world.size()) {
            let cell = self.grid.get_checked_mut(coord);
            cell.last_seen = self.count;
            cell.last_lit = self.count;
            cell.visible_directions = DirectionBitmap::all();
            cell.light_colour = ambient_light_colour;
            f(&mut cell.data, coord);
        }
        self.apply_lights(world);
    }

    pub fn get_visibility(&self, coord: Coord) -> CellVisibility<&T> {
        if let Some(cell) = self.grid.get(coord) {
            cell.visibility(self.count)
        } else {
            CellVisibility::Never
        }
    }

    pub fn get_data(&self, coord: Coord) -> Option<&T> {
        self.get_visibility(coord).into_data()
    }

    pub fn enumerate(&self) -> impl Iterator<Item = (Coord, CellVisibility<&T>)> {
        self.grid
            .enumerate()
            .map(|(coord, cell)| (coord, cell.visibility(self.count)))
    }
}
