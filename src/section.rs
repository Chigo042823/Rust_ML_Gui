use graphics::{rectangle, Context};
use opengl_graphics::GlGraphics;

use crate::widget::{Widget, WidgetType};


const OUTLINE: [f32; 4] = [0.6, 0.6, 0.6, 1.0];
const PADDING: f64 = 10.0;
pub struct Section {
    pub coords: [f64; 4],
    pub width: f64,
    pub height: f64,
    pub widgets: Vec<Widget>,
    pub cost: Vec<f64>
}

impl Section {
    pub fn new(coords: [f64; 4], width: f64, height: f64, cost: Vec<f64>) -> Self {
        Section {
            coords,
            width, 
            height,
            widgets: vec![],
            cost
        }
    }

    pub fn set_widgets(&mut self, widget_count: usize) {
        let widget_height: f64 = (self.height / widget_count as f64) - PADDING;
        let widget_width = self.width - (PADDING * 2.0);

        let mut widgts = vec![];
        for i in 0..widget_count {
            let coords1 = [self.coords[0], self.coords[1] + (widget_height * i as f64)];
            let coords = [coords1[0] + PADDING,
                coords1[1] + PADDING,
                coords1[0] - PADDING,
                coords1[1] - PADDING
            ];
            widgts.push(Widget::new(coords, widget_width, widget_height, WidgetType::CostPlot, self.cost.clone()));
        }
        self.widgets = widgts;
    }

    pub fn render(&mut self, ctx: Context, gl: &mut GlGraphics) {
        let rect = rectangle::rectangle_by_corners(self.coords[0], self.coords[1],
                    self.coords[2] + self.width, self.coords[3] + self.height);

        rectangle::Rectangle::new_border(OUTLINE, 1.0)
            .draw(rect, &ctx.draw_state, ctx.transform, gl);

        for i in 0..self.widgets.len() {
            self.widgets[i].render(ctx, gl);
        }
    }
}