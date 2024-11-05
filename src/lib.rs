use direction::DirectionBitmap;
pub use grid_2d::GridEnumerate;
use grid_2d::{Coord, CoordIter, Grid, Size};
use rgb_int::Rgb24;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
pub use shadowcast::{vision_distance, VisionDistance};
use shadowcast::{Context as ShadowcastContext, InputGrid};
use std::marker::PhantomData;

/// Describes how a light diminishes with distance from the light source
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct Diminish {
    /// The height of the light off the ground. Higher lights will spread out further. The default
    /// is 1.0.
    pub height: f64,
    /// How tightly the pool of light is focused. A value of 0 is completely unfocused, such as a
    /// light that emits rays evenly in all directions. Higher values produce a more concentrated
    /// pool of light that spreads out less far. Negative values produce a saturated region at the
    /// centre of the pool of light, growing with the magnitude of the negative value. The default
    /// is 0.
    pub focus: f64,
    /// The intensity of the light directly under the light source. Note that lighting calculations
    /// are normalized so that increasing the height doesn't decrease the intensity of the light
    /// under the light source. The default is 1.0.
    pub intensity: f64,
}

impl Default for Diminish {
    fn default() -> Self {
        Self {
            height: 1.0,
            focus: 0.0,
            intensity: 1.0,
        }
    }
}

impl Diminish {
    pub fn with_height(self, height: f64) -> Self {
        Self { height, ..self }
    }

    pub fn with_focus(self, focus: f64) -> Self {
        Self { focus, ..self }
    }

    pub fn with_intensity(self, intensity: f64) -> Self {
        Self { intensity, ..self }
    }

    // `intensity` is just a scaling factor for the result, so it will be ignored in this
    // explanation. Thus the intensity of the light at a distance of 0 will be 1.
    //
    // We'll start by ignoring `focus` for simplicity and add it in later. Start with the formula:
    // `1 / (d^2 + h^2)` which is the inverse of the square of the 3d distance from the light
    // source. To normalize it such that the intensity at a distance of 0 is 1, multiply the entire
    // thing by h^2. On its own this function produces a pool of light which is unfocused. This
    // function reduces slowly around d=0, then faster, before slowing down as its value approaches
    // 0.
    //
    // To add the focus, first consider `1 / (d + h)^2` (where h is the constant height). Observe
    // that this is the same shape as `1 / d^2`, but that it's shifted to the left such that it
    // crosses d=0 at `1 / h^2`. `1 / (d + h)^2` drops quickly from d=0 and slows down as its value
    // approaches 1, so it's tightly focussed. The goal of the focus parameter is to interpolate
    // between the unfocused and focused functions. This is simple, because the denominator:
    // `(d + h)^2 == d^2 + 2dh + h^2`, which is equivalent to the denominator of the unfocused
    // formula with the addition of 2dh. Thus by introducing a focus parameter `f` we can
    // interpolate between the two formulae with:
    // `h^2 / (d^2 + 2dhf + h^2)`.
    fn intensity_at_distance(&self, distance: f64) -> f64 {
        (self.intensity * self.height * self.height)
            / ((distance * distance)
                + (2.0 * self.focus * self.height * distance)
                + (self.height * self.height))
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct Light<V: VisionDistance> {
    pub colour: Rgb24,
    pub vision_distance: V,
    pub diminish: Diminish,
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
                        let distance = ((light_coord - cell_coord).magnitude2() as f64).sqrt();
                        let light_colour = light.colour.saturating_scalar_mul_f64(
                            light.diminish.intensity_at_distance(distance),
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
