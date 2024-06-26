extern crate image;

use graphics::{rectangle, Context};
use piston_window::*;

use crate::widget::{Widget, WidgetType};


const OUTLINE: [f32; 4] = [0.6, 0.6, 0.6, 1.0];
pub struct Section {
    pub coords: [f64; 4],
    pub width: f64,
    pub height: f64,
    pub widgets: Vec<Widget>,
    pub cost: f64,
    pub padding: [f64; 2]
}

impl Section {
    pub fn new(coords: [f64; 4], width: f64, height: f64) -> Self {
        Section {
            coords,
            width, 
            height,
            widgets: vec![],
            cost: 0.0,
            padding: [width * 0.05, height * 0.05],
        }
    }

    pub fn set_widgets(&mut self, widgets: &Vec<WidgetType>) {
        let widget_count = widgets.len();
        let widget_height: f64 = (self.height / widget_count as f64) - (self.padding[1] * 1.0);
        let widget_width = self.width - (self.padding[0] * 2.0);

        let mut widgts = vec![];
        for i in 0..widget_count {
            let coords1 = [self.coords[0], self.coords[1] + (widget_height * i as f64)];
            let coords = [coords1[0] + self.padding[0],
                coords1[1] + self.padding[1],
                coords1[0] - self.padding[0],
                coords1[1] - self.padding[1]
            ];
            widgts.push(Widget::new(coords, widget_width, widget_height, widgets[i].clone()));
        }
        self.widgets = widgts;
    }

    pub fn render(&mut self, ctx: Context, gl: &mut G2d, window_ctx: &mut G2dTextureContext) {
        let rect = rectangle::rectangle_by_corners(self.coords[0], self.coords[1],
                    self.coords[2] + self.width, self.coords[3] + self.height);

        rectangle::Rectangle::new_border(OUTLINE, 1.0)
            .draw(rect, &ctx.draw_state, ctx.transform, gl);

        for i in 0..self.widgets.len() {
            self.widgets[i].render(ctx, gl, window_ctx);
        }
    }
    
    pub fn update(&mut self, 
        cost: f64, 
        epochs: usize, 
        nn_data: Vec<Vec<f64>>,
        layer_data: ((Vec<Vec<Vec<Vec<f64>>>>, Vec<Vec<Vec<f64>>>), (Vec<f64>, Vec<Vec<f64>>), Vec<usize>, Vec<Vec<Vec<Vec<f64>>>>),
    ) {
        for i in 0..self.widgets.len() {
            let widget = &mut self.widgets[i];
            let mut layers= ((vec![], vec![]), (vec![], vec![]), vec![], vec![]);
            if widget.widget_type == WidgetType::Architecture ||
                widget.widget_type == WidgetType::ConvArch {
                layers = layer_data.clone();
            }
            widget.update(
                cost, 
                layers, 
                epochs,
                nn_data.clone()
            );
        }
    }
}