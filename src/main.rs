use ml_gui::{gui::GUI, widget::WidgetType};
use ml_library::{layer::Layer, layer::LayerType::*, loss_function::LossType::*, network::Network, conv_params::PaddingType::*, activation::ActivationFunction::*};
use image::*;
use WidgetType::*;

fn main() {
    

    // nn.load_model("assets/models/cnnTanH");
    // conv_digit_test();
    conv_digit_model();
    // xor_model()
    // dense_digit_model(app);
    // softmax_test();
}

pub fn xor_model() {

    let layers: Vec<Layer> = vec![
        Layer::dense([2, 3], Sigmoid),
        Layer::dense([3, 2], Sigmoid),
    ];

    let mut nn = Network::new(layers, 0.5, 2, MSE);
    let sections: Vec<Vec<WidgetType>> = vec![
        vec![CostPlot], 
        vec![Architecture]
    ];

    let mut app = GUI::new(nn);
    app.set_sections(sections);
    app.set_epochs_per_second(20);
    app.set_cost_expiration(false, 20);
    app.set_model_name("assets/models/XOR");

    let dense_data: Vec<[Vec<f64>; 2]> = vec![
        [vec![1.0, 0.0], vec![0.0, 1.0]],
        [vec![0.0, 0.0], vec![1.0, 0.0]],
        [vec![1.0, 1.0], vec![1.0, 0.0]],
        [vec![0.0, 1.0], vec![0.0, 1.0]],
    ]; 

    app.set_dense_data(dense_data);
    app.run();
}

pub fn dense_digit_model(mut app: GUI) {
    let mut dense_data: Vec<[Vec<f64>; 2]> = vec![];

    let img = image::open("assets/img/mnist/mnist_8.png").unwrap();

    let dims = img.dimensions();

    for y in 0..dims.1 {
        for x in 0..dims.0 {
            let pixel = img.get_pixel(x, y).0;
            let intensity = (pixel[0] / 3) + (pixel[1] / 3) + (pixel[2] / 3);
            let x_coord = x as f64 / (dims.0 - 1) as f64;
            let y_coord = y as f64 / (dims.1 - 1) as f64;
            let inputs = vec![x_coord, y_coord];
            dense_data.push([inputs, vec![(intensity as f64 / 255.0)]]);
        }
    }

    app.set_dense_data(dense_data);
    app.run();
}

pub fn conv_digit_test() {

    let mut nn = Network::from_load("assets/models/cnnTest");

    let num = 8;
    let img = image::open(format!("assets/img/mnist/tMnist_{}.png", num)).unwrap();

    let dims = img.dimensions();

    let mut inputs = vec![];

    for y in 0..dims.1 {
        let mut row = vec![];
        for x in 0..dims.0 {
            let pixel = img.get_pixel(x, y).0;
            let intensity = (pixel[0] / 3) + (pixel[1] / 3) + (pixel[2] / 3);
            row.push(intensity as f64);
        }
        inputs.push(row);
    }
    println!("{}: {:?}", num, nn.conv_forward(vec![inputs]).iter()
        .map(|x| (*x * 1_000_000.0) as u32)
        .enumerate()
        .max_by_key(|&(_, value)| value)
        .map(|(index, _)| index + 1).unwrap()
    );
}

pub fn conv_digit_model() {
    let layers: Vec<Layer> = vec![
        Layer::conv(3, Valid, 1, ReLU),
        Layer::pool(2, 2),
        Layer::pool(3, 2),
        Layer::dense([36, 32], Sigmoid),
        Layer::dense([32, 10], SoftMax),
    ];

    let mut nn = Network::new(layers, 0.002, 6, CEL);
    let sections: Vec<Vec<WidgetType>> = vec![
        vec![ConvArch], 
    ];

    let mut app = GUI::new(nn);
    app.set_sections(sections);
    app.set_epochs_per_second(20);
    app.set_cost_expiration(false, 20);
    app.set_model_name("assets/models/cnnTest");

    let mut conv_data: Vec<(Vec<Vec<Vec<f64>>>, Vec<f64>)> = vec![];

    let max_nums = app.nn.get_nodes().last().unwrap().clone();
    for i in 0..max_nums {
        let img = image::open(format!("assets/img/mnist/mnist_{}.png", i)).unwrap();

        let dims = img.dimensions();

        let mut inputs = vec![];

        for y in 0..dims.1 {
            let mut row = vec![];
            for x in 0..dims.0 {
                let pixel = img.get_pixel(x, y).0;
                let intensity = (pixel[0] / 3) + (pixel[1] / 3) + (pixel[2] / 3);
                row.push(intensity as f64 / 255.0);
            }
            inputs.push(row);
        }
        let mut output = vec![0.0; max_nums];
        output[i] = 1.0;
        conv_data.push((vec![inputs], output));
    }

    app.set_conv_data(conv_data);
    app.run();
}

pub fn softmax_test() {
    let layers: Vec<Layer> = vec![
        Layer::conv(3, Same, 1, ReLU),
        Layer::dense([9, 6], TanH),
        Layer::dense([6, 3], SoftMax),
    ];

    let nn = Network::new(layers, 0.02, 2, CEL);
    let sections: Vec<Vec<WidgetType>> = vec![
        vec![CostPlot], 
    ];

    let mut app = GUI::new(nn);
    app.set_sections(sections);
    app.set_epochs_per_second(1);
    app.set_cost_expiration(false, 20);
    app.set_model_name("assets/models/cnnTest");

    let conv_data: Vec<(Vec<Vec<Vec<f64>>>, Vec<f64>)> = vec![
        (
            vec![
                vec![
                    vec![1.0, 0.0, 1.0],
                    vec![0.0, 1.0, 0.0],
                    vec![1.0, 0.0, 1.0],
                ]
            ],
            vec![1.0, 0.0, 0.0]
        ),
        (
            vec![
                vec![
                    vec![0.0, 1.0, 0.0],
                    vec![1.0, 1.0, 1.0],
                    vec![0.0, 1.0, 0.0],
                ]
            ],
            vec![0.0, 1.0, 0.0]
        ),
        (
            vec![
                vec![
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 0.0],
                ]
            ],
            vec![0.0, 0.0, 1.0]
        ),
    ];

    app.set_conv_data(conv_data);
    app.run();
}

pub fn sin_model(app: &mut GUI) {

    app.x_range = [-10.0, 10.0];

    let mut func_outputs = vec![];
    let increment = 0.1;
    let mut counter = app.x_range[0];

    while counter < app.x_range[1] {
        func_outputs.push([vec![counter.clone()], vec![func(&counter)]]);
        counter += increment;
    }

    app.set_dense_data(func_outputs);
    app.run()
}

fn func(x: &f64) -> f64 {
    x.sin()
}

fn upscale_img(nn: &mut Network, dims: [u32; 2]) {

    // let output_image = ImageBuffer::from_fn(dims[0], dims[1], |x, y| {
    //     let x_coord = x as f64 / (dims[0] - 1) as f64;
    //     let y_coord = y as f64 / (dims[1] - 1) as f64;
    //     let inputs = vec![x_coord, y_coord];
    //     let pix = (nn.forward(inputs)[0] * 255.0) as u8;
    //     Rgba([pix, pix, pix, 255]) // Varying colors in a gradient
    // });
            

    // let _  = output_image.save("Output.png").unwrap();
    // println!("Saved Image");
}