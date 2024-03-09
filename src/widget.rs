use graphics::*;
use opengl_graphics::GlGraphics;

const OUTLINE: [f32; 4] = [0.6, 0.6, 0.6, 1.0];
const PADDING: f64 = 20.0;
const LINE_THICKNESS: f64 = 0.7;
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

pub enum WidgetType {
    CostPlot
}

pub struct Widget {
    pub coords: [f64; 4],
    pub width: f64,
    pub height: f64,
    pub widget_type: WidgetType, 
    pub cost: Vec<f64>
}

impl Widget {
    pub fn new(coords: [f64; 4], width: f64, height: f64, widget_type: WidgetType, cost: Vec<f64>) -> Self {
        Widget {
            coords,
            width: width, 
            height: height,
            widget_type,
            cost
        }
    }

    pub fn render(&mut self, ctx: Context, gl: &mut GlGraphics) {
        let rect = rectangle::rectangle_by_corners(self.coords[0], self.coords[1],
            self.coords[2] + self.width, self.coords[3] + self.height);

        rectangle::Rectangle::new_border(OUTLINE, LINE_THICKNESS)
            .draw(rect, &ctx.draw_state, ctx.transform, gl);
        match self.widget_type {
            WidgetType::CostPlot => self.draw_costplot(ctx, gl, self.cost.clone()),
        }
    }

    pub fn draw_costplot(&mut self, ctx: Context, gl: &mut GlGraphics, cost: Vec<f64>) {
        let floor = self.coords[1] + self.height - PADDING * 2.0;
        let wall = self.coords[0] + PADDING;
        let min_coord = [wall, floor];
        let y_max_coord = [wall, self.coords[1] + PADDING];
        let x_max_coord = [self.coords[0] + self.width - PADDING * 2.0, floor];
        line_from_to(OUTLINE, LINE_THICKNESS, y_max_coord, min_coord, ctx.transform, gl);
        line_from_to(OUTLINE, LINE_THICKNESS, x_max_coord, min_coord, ctx.transform, gl);

        let mut last_point = y_max_coord.clone();
        let cost_count = cost.len() as f64;
        for i in 0..cost_count as usize{
            //TODO fix x coordinate calculation
            let next_point = [(x_max_coord[0] / cost_count), (y_max_coord[1] / cost_count) * self.cost[i]];
            line_from_to(RED, LINE_THICKNESS, last_point, next_point, ctx.transform, gl);
            last_point = next_point;
        }
    }
}