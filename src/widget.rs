use std::{f64::INFINITY, process::Output, usize};

use graphics::{rectangle::rectangle_by_corners, *};
use ml_library::layer::Layer;
use opengl_graphics::{GlGraphics, GlyphCache, TextureSettings};
use rand::Rng;

const OUTLINE: [f32; 4] = [0.7, 0.7, 0.7, 1.0];
const PADDING: f64 = 20.0;
const LINE_THICKNESS: f64 = 0.7;

#[derive(Clone)]
pub enum WidgetType {
    CostPlot,
    Architecture
}

pub struct Widget {
    pub coords: [f64; 4],
    pub width: f64,
    pub height: f64,
    pub widget_type: WidgetType, 
    pub cost: Vec<f64>,
    pub layers: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, Vec<usize>)
}

impl Widget {
    pub fn new(coords: [f64; 4], width: f64, height: f64, widget_type: WidgetType) -> Self {
        Widget {
            coords,
            width: width, 
            height: height,
            widget_type,
            cost: vec![],
            layers: (vec![], vec![], vec![])
        }
    }

    pub fn render(&mut self, ctx: Context, gl: &mut GlGraphics) {
        let rect = rectangle::rectangle_by_corners(self.coords[0], self.coords[1],
            self.coords[2] + self.width, self.coords[3] + self.height);

        rectangle::Rectangle::new_border(OUTLINE, LINE_THICKNESS)
            .draw(rect, &ctx.draw_state, ctx.transform, gl);
        match self.widget_type {
            WidgetType::CostPlot => self.draw_costplot(ctx, gl, self.cost.clone()),
            WidgetType::Architecture => self.draw_architecture(ctx, gl),
        }
    }

    pub fn update(&mut self, gl: &mut GlGraphics, cost: f64, layers: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, Vec<usize>)) {  
        self.cost.push(cost);
        self.layers = layers;
    }

    pub fn draw_costplot(&mut self, ctx: Context, gl: &mut GlGraphics, cost: Vec<f64>) {
        let floor = self.coords[1] + self.height - PADDING * 2.0;
        let wall = self.coords[0] + PADDING;
        let min_coord = [wall, floor];
        let y_max_coord = [wall, self.coords[1] + (PADDING * 2.0)];
        let x_max_coord = [self.coords[0] + self.width - PADDING * 2.0, floor];
        line_from_to(OUTLINE, LINE_THICKNESS, y_max_coord, min_coord, ctx.transform, gl);
        line_from_to(OUTLINE, LINE_THICKNESS, x_max_coord, min_coord, ctx.transform, gl);

        let cost_count = cost.len() as f64;

        if cost_count == 0.0 {
            return;
        }

        let max_cost = Self::get_max_cost(&cost);

        let y =  x_max_coord[1] + ((cost[0] / max_cost) * (y_max_coord[1] - x_max_coord[1]));

        let mut last_point = [wall, y];

        let mut line_color: [f32; 4] = [0.0, 0.8, 0.0, 1.0];

        let mut glyphs = GlyphCache::new("assets/BebasNeue-Regular.ttf", (), TextureSettings::new()).unwrap();

        let cost_text = "Cost: ".to_owned() + &cost[cost.len() - 1].to_string();

        let epochs_text: String = "Epochs: ".to_owned() + &(cost.len() * 10).to_string();

        let _ = text::Text::new_color(OUTLINE, 20).draw_pos(&cost_text,
            [y_max_coord[0], y_max_coord[1] - PADDING / 2.0], &mut glyphs, &ctx.draw_state, ctx.transform, gl);

        let _ = text::Text::new_color(OUTLINE, 20).draw_pos(&epochs_text,
                [self.width / 2.0, y_max_coord[1] - PADDING / 2.0], &mut glyphs, &ctx.draw_state, ctx.transform, gl);

        for i in 0..cost_count as usize{
            //TODO fix x coordinate calculation
            line_color[0] = (cost[i] / max_cost) as f32;
            line_color[1] = 1.0 - (cost[i] / max_cost) as f32;
            let next_x = y_max_coord[0] + (((i + 1) as f64 / cost_count) * (x_max_coord[0] - y_max_coord[0]));
            let next_y =  x_max_coord[1] + ((cost[i] / max_cost) * (y_max_coord[1] - x_max_coord[1]));
            let next_point = [next_x, next_y];
            line_from_to(line_color, LINE_THICKNESS, last_point, next_point, ctx.transform, gl);
            last_point = next_point.clone();
        }
    }

    pub fn draw_architecture(&mut self, ctx: Context, gl: &mut GlGraphics) {

        let layer_nodes = self.layers.2.clone();
        let weights = self.layers.0.clone();
        let biases = self.layers.1.clone();

        let mut neuron_color = OUTLINE;
        let mut weight_color = OUTLINE;

        let floor = self.coords[1] + self.height - PADDING * 2.0;
        let wall = self.coords[0] - PADDING;
        let y_center = floor - ((self.height / 2.0) - (PADDING * 4.0));
        let x_center = wall + (self.width / 2.0) + (PADDING * 2.0);

        let max_nodes = Self::get_max_nodes(&layer_nodes) as f64;
        let neuron_size = PADDING;
        let layer_width = (self.width / layer_nodes.len() as f64) - neuron_size - PADDING;
        let network_width = layer_width * layer_nodes.len() as f64;

        for i in 0..layer_nodes.len() { //layers
            let layer_height =  ((self.height - (PADDING * 2.0)) - neuron_size * 2.0) * (layer_nodes[i] as f64 / max_nodes);

            let neuron_spacing = (layer_height / layer_nodes[i] as f64) - (neuron_size * 2.0);

            let x = x_center - (network_width / 2.0) + (i as f64 * layer_width) + (neuron_size * 2.0);

            for j in 0..layer_nodes[i] { //layer nodes
                    if i != 0 {
                        neuron_color[0] = Self::sigmoid(biases[i - 1][j]);
                        neuron_color[1] = 1.0 - Self::sigmoid(biases[i - 1][j]);
                        neuron_color[2] = Self::sigmoid(biases[i - 1][j]) / 1.0;
                    }
                    let y = y_center - (layer_height / 2.0) + (j as f64 * ((neuron_size * 2.0) + neuron_spacing));
                    let rect = rectangle_by_corners(x - neuron_size, y - neuron_size, x + neuron_size, y + neuron_size);
                    ellipse::Ellipse::new(neuron_color).draw(rect, &ctx.draw_state, ctx.transform, gl);

                if i + 1 != layer_nodes.len() {
                    let next_layer_height = ((self.height - (PADDING * 2.0)) - neuron_size * 2.0) * (layer_nodes[i + 1] as f64 / max_nodes);
                    for k in 0..layer_nodes[i + 1] { //next nodes
                        weight_color[0] = Self::sigmoid(weights[i][j][k]);
                        weight_color[1] = 1.0 - Self::sigmoid(weights[i][j][k]);
                        weight_color[2] = Self::sigmoid(weights[i][j][k]) / 1.0;
                        let next_neuron_spacing = (next_layer_height / layer_nodes[i + 1] as f64) - (neuron_size * 2.0);
                        let next_x = x_center - (network_width / 2.0) + ((i + 1) as f64 * layer_width) + (neuron_size * 2.0);
                        let next_y = y_center - (next_layer_height / 2.0) + (k as f64 * ((neuron_size * 2.0) + next_neuron_spacing));

                        line_from_to(weight_color, LINE_THICKNESS, 
                            [x, y], 
                            [next_x, next_y], 
                            ctx.transform, 
                            gl);
                    }
                }
            }
        }
    }

    fn get_max_nodes(nodes: &Vec<usize>) -> usize {
        let mut max = 0;
        for i in 0..nodes.len() {
            if nodes[i] > max {
                max = nodes[i];
            }
        }
        max
    }

    fn sigmoid(x: f64) -> f32 {
        1.0 / (1.0 + (-x).exp()) as f32
    }

    fn get_max_cost(cost: &Vec<f64>) -> f64 {
        let mut max = -INFINITY;
        for i in 0..cost.len() {
            if cost[i] > max {
                max = cost[i];
            }
        }
        max
    }
}