extern crate image;
extern crate opengl_graphics;

use std::f64::INFINITY;

use graphics::{rectangle::{self, rectangle_by_corners}, Context};
use image::{ImageBuffer, Rgba};
use piston_window::*;
use WidgetType::*;

const OUTLINE: [f32; 4] = [0.7, 0.7, 0.7, 1.0];
const LINE_THICKNESS: f64 = 0.7;

#[derive(Clone, PartialEq)]
pub enum WidgetType {
    CostPlot,
    Architecture,
    OutputImg,
    OutputGraph
}

pub struct Widget {
    pub coords: [f64; 4],
    pub width: f64,
    pub height: f64,
    pub widget_type: WidgetType, 
    pub cost: Vec<f64>,
    pub layers: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, Vec<usize>),
    pub epochs: usize,
    pub cost_expiration: bool,
    pub cost_expiration_epochs: usize,
    pub expected_dense_data: Vec<[Vec<f64>; 2]>,
    pub expected_conv_data: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    pub max_expected_dense_data: f64,
    pub nn_data: Vec<Vec<f64>>,
    pub max_nn_data: u64,
    pub padding: [f64; 2]
}

impl Widget {
    pub fn new(coords: [f64; 4], width: f64, height: f64, widget_type: WidgetType) -> Self {
        Widget {
            coords,
            width: width, 
            height: height,
            widget_type,
            cost: vec![],
            layers: (vec![], vec![], vec![]),
            epochs: 0,
            cost_expiration: false,
            cost_expiration_epochs: 0,
            expected_dense_data: vec![],
            expected_conv_data: vec![],
            max_expected_dense_data: 0.0,
            nn_data: vec![],
            max_nn_data: 0,
            padding: [width *0.05, height * 0.05]
        }
    }

    pub fn render(&mut self, ctx: Context, gl: &mut G2d, window_ctx: &mut G2dTextureContext) {
        let rect = rectangle::rectangle_by_corners(self.coords[0], self.coords[1],
            self.coords[2] + self.width, self.coords[3] + self.height);

        rectangle::Rectangle::new_border(OUTLINE, LINE_THICKNESS)
            .draw(rect, &ctx.draw_state, ctx.transform, gl);
        match self.widget_type {
            CostPlot => self.draw_costplot(ctx, gl),
            Architecture => self.draw_architecture(ctx, gl),
            OutputImg => self.draw_image(ctx, gl, window_ctx),
            OutputGraph => self.draw_output_graph(ctx, gl)
        }   
    }

    pub fn set_dense_data(&mut self, dense_data: Vec<[Vec<f64>; 2]>) {
        self.expected_dense_data = dense_data;
        self.max_expected_dense_data = Self::get_max_output(&self.expected_dense_data);
    }

    pub fn set_conv_data(&mut self, conv_data: Vec<(Vec<Vec<f64>>, Vec<f64>)>) {
        self.expected_conv_data = conv_data;
    }

    pub fn set_cost_expiration(&mut self, expire: bool, epochs: usize) {
        self.cost_expiration = expire;
        self.cost_expiration_epochs = epochs;
    }

    pub fn pop_cost(&mut self) {
        let chunk = 100;
        if self.cost.len() > 0 && self.cost.len() + chunk > chunk * 2 {
            for _ in 0..chunk {
                self.cost.remove(0);
            }
        }
    }

    pub fn update(&mut self, 
        cost: f64, 
        layers: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, Vec<usize>),
        epochs: usize,
        nn_data: Vec<Vec<f64>>
    ) {  
        match self.widget_type {
            CostPlot => 
                self.cost.push(cost),
            _ => ()
        }
        self.layers = layers;
        self.epochs += epochs;
        self.nn_data = nn_data;
        // self.max_nn_data = self.nn_data.iter().map(|x| (x[0] * 100_000.0) as u64).max().unwrap();

        if self.epochs % (self.cost_expiration_epochs + 1) == 0 && self.cost_expiration {
            if self.cost.len() != 0 {
                self.cost.remove(0);
            }
        }
    }

    pub fn draw_image(&mut self, ctx: Context, gl: &mut G2d, window_ctx: &mut G2dTextureContext) {

        if self.expected_dense_data.len() == 0 as usize || self.nn_data.len() == 0 as usize {
            return;
        }

        let expected_image = &self.expected_dense_data;
        let nn_image = &self.nn_data;

        let (w, h) = (65, 67);
        let mut img_i = 0;

        let base_image = ImageBuffer::from_fn(w, h, |x, y| {
            let pix = (expected_image[img_i][1][0] * 255.0) as u8;
            img_i += 1;
            Rgba([pix, pix, pix, 255]) // Varying colors in a gradient
        });

        img_i = 0;

        let output_image = ImageBuffer::from_fn(w, h, |x, y| {
            let pix = (nn_image[img_i][0] * 255.0) as u8;
            img_i += 1;
            Rgba([pix, pix, pix, 255]) // Varying colors in a gradient
        });
        
        // Create a texture from the image dense_data
        let output_texture = piston_window::Texture::from_image(
            window_ctx,
            &output_image,
            &TextureSettings::new(),
        ).unwrap();

        // Create a texture from the image dense_data
        let base_texture = piston_window::Texture::from_image(
            window_ctx,
            &base_image,
            &TextureSettings::new(),
        ).unwrap();

        let w_scale = (((self.width * 1.5 - self.padding[0] * 2.0) / 2.0) / w as f64) / 2.0;
        let h_scale = ((self.height * 1.5 - self.padding[1] * 2.0) / 2.0) / h as f64;

        let x = self.coords[0] + (self.width / 2.0) - (w_scale * w as f64) - self.padding[0];
        let y = (self.coords[1] + (self.height / 2.0)) - (h_scale * h as f64) / 1.5;

        piston_window::image(&output_texture, ctx.transform.trans(x + (w_scale * w as f64), y).scale(w_scale, h_scale), gl);
        piston_window::image(&base_texture, ctx.transform.trans(x, y).scale(w_scale, h_scale), gl);
    }

    pub fn draw_output_graph(&mut self, ctx: Context, gl: &mut G2d) 
    {

        if self.expected_dense_data.len() == 0 as usize || self.nn_data.len() == 0 as usize {
            return;
        }
        
        let floor = self.coords[3] + self.height - self.padding[1];
        let wall = self.coords[0] + self.padding[0];
        let min_coord = [wall, floor];
        let y_max_coord = [wall, self.coords[1] + (self.padding[1] * 2.0)];
        let x_max_coord = [self.coords[2] + self.width - self.padding[0], floor];
        line_from_to(OUTLINE, LINE_THICKNESS, y_max_coord, min_coord, ctx.transform, gl);
        line_from_to(OUTLINE, LINE_THICKNESS, x_max_coord, min_coord, ctx.transform, gl);

        let y_range = [-1.0, 1.0];
        let x_range = [-10.0, 10.0];
        let increment = 0.1;
        let mut counter = x_range[0];
        let mut index = 0;

        let expected_color: [f32; 4] = [0.0, 0.8, 0.0, 1.0];
        let nn_color: [f32; 4] = [0.0, 0.0, 0.8, 1.0];

        counter = x_range[0];

        let expected_dense_data = &self.expected_dense_data;
        let nn_data = &self.nn_data;

        let max_expected_output = self.max_expected_dense_data + 1.0;
        
        let expected_y = (expected_dense_data[0][1][0] + 1.0) / max_expected_output;
        let network_y = (nn_data[0][0] + 1.0) / max_expected_output;

        let mut last_expected_point = [y_max_coord[0], expected_y];
        let mut last_network_point = [y_max_coord[0], network_y];

        while counter < x_range[1] {
            let next_expected_x = y_max_coord[0] + ((counter / x_range[1]) * (x_max_coord[0] - y_max_coord[0]));
            let next_expected_y =  x_max_coord[1] - ((expected_dense_data[index][1][0] + 1.0) / max_expected_output) * (x_max_coord[1] - y_max_coord[1]);
            let next_expected_point = [next_expected_x, next_expected_y];
            if (next_expected_y < floor && next_expected_y > y_max_coord[1]) &&
                (next_expected_x > wall && next_expected_x < x_max_coord[0])
            {
                line_from_to(expected_color, LINE_THICKNESS, last_expected_point, next_expected_point, ctx.transform, gl);
            }
            last_expected_point = next_expected_point;

            let next_network_x = y_max_coord[0] + ((counter / x_range[1]) * (x_max_coord[0] - y_max_coord[0]));
            let next_network_y =  x_max_coord[1] - ((nn_data[index][0] + 1.0) / max_expected_output) * (x_max_coord[1] - y_max_coord[1]);
            let next_network_point = [next_network_x, next_network_y];
            if (next_network_y < floor && next_network_y > y_max_coord[1]) &&
                (next_network_x > wall && next_network_x < x_max_coord[0])
            {
                line_from_to(nn_color, LINE_THICKNESS, last_network_point, next_network_point, ctx.transform, gl);
            }
            last_network_point = next_network_point;

            counter += increment;
            index += 1;
        }
    }

    pub fn draw_costplot(&mut self, ctx: Context, gl: &mut G2d) {
        let cost = &self.cost;

        let floor = self.coords[3] + self.height - self.padding[1];
        let wall = self.coords[0] + self.padding[0];
        let min_coord = [wall, floor];
        let y_max_coord = [wall, self.coords[1] + (self.padding[1] * 2.0)];
        let x_max_coord = [self.coords[2] + self.width - self.padding[0], floor];
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

        for i in 1..cost_count as usize{
            //TODO fix x coordinate calculation
            line_color[0] = (cost[i] / max_cost) as f32;
            line_color[1] = 1.0 - (cost[i] / max_cost) as f32;
            let next_x = y_max_coord[0] + (((i + 1) as f64 / cost_count) * (x_max_coord[0] - y_max_coord[0]));
            let next_y =  x_max_coord[1] + ((cost[i] / max_cost) * (y_max_coord[1] - x_max_coord[1]));
            let next_point = [next_x, next_y];
            line_from_to(line_color, LINE_THICKNESS, last_point, next_point, ctx.transform, gl);
            last_point = next_point;
        }
    }

    pub fn draw_architecture(&mut self, ctx: Context, gl: &mut G2d) {

        let layer_nodes = &self.layers.2;
        let weights = &self.layers.0;
        let biases = &self.layers.1;

        let mut neuron_color = [0.7, 0.7, 0.0, 1.0];
        let mut weight_color = [0.7, 0.7, 0.0, 1.0];

        let floor = self.coords[1] + self.height - (self.padding[1] * 0.5);
        let wall = self.coords[0] + (self.padding[0] * 2.0);
        let y_center = floor - (self.height / 2.0);
        let x_center = wall + (self.width / 2.0);

        let max_nodes = Self::get_max_nodes(&layer_nodes) as f64;
        let neuron_size =  50.0 * ((((self.width / ctx.get_view_size()[0]) + (self.height / ctx.get_view_size()[1])) / 2.0) / max_nodes);
        let network_width = self.width +  -(self.padding[0] * 2.0) + -(neuron_size * 2.0);
        let layer_width = network_width / layer_nodes.len() as f64;

        for i in 0..layer_nodes.len() { //layers
            let layer_height =  ((self.height - (self.padding[1] * 2.0)) - (layer_nodes[i] as f64 * neuron_size)) * (layer_nodes[i] as f64 / max_nodes);

            let neuron_spacing = layer_height / layer_nodes[i] as f64;

            // let x = x_center - (network_width / 2.0) + (i as f64 * layer_width) + (neuron_size * 2.0);
            let x = (x_center - (network_width / 2.0)) + (i as f64 * layer_width);

            for j in 0..layer_nodes[i] { //layer nodes
                    if i != 0 {
                        let val = Self::sigmoid(biases[i - 1][j]);
                        neuron_color[0] = 1.0 - val;
                        neuron_color[1] = val;
                        neuron_color[2] = 1.0 - val;
                    }
                    // let y = y_center - (layer_height / 2.0) + (j as f64 * ((neuron_size * 2.0) + neuron_spacing));
                    let y = (y_center - (layer_height / 2.0)) + (neuron_size * 2.0) + (j as f64 * neuron_spacing);
                    let rect = rectangle_by_corners(x - neuron_size, y - neuron_size, x + neuron_size, y + neuron_size);
                    ellipse::Ellipse::new(neuron_color).draw(rect, &ctx.draw_state, ctx.transform, gl);

                if i + 1 != layer_nodes.len() {
                    let next_layer_height = ((self.height - (self.padding[1] * 2.0)) - (layer_nodes[i + 1] as f64 * neuron_size)) * (layer_nodes[i + 1] as f64 / max_nodes);
                    for k in 0..layer_nodes[i + 1] { //next nodes
                        let val = Self::sigmoid(weights[i][j][k]);
                        weight_color[0] = 1.0 - val;
                        weight_color[1] = val;
                        weight_color[2] = 1.0 - val;
                        let next_neuron_spacing = next_layer_height / layer_nodes[i + 1] as f64;
                        let next_x = (x_center - (network_width / 2.0)) + ((i + 1) as f64 * layer_width);
                        let next_y = (y_center - (next_layer_height / 2.0)) + (neuron_size * 2.0) + (k as f64 * next_neuron_spacing);

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

    fn get_max_output(outputs: &Vec<[Vec<f64>; 2]>) -> f64 {
        let mut max = -INFINITY;
        for i in 0..outputs.len() {
            for j in 0..outputs[i][1].len() { 
                if outputs[i][1][j] > max {
                    max = outputs[i][1][j];
                }
            }
        }
        max
    }
}