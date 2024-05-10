async function loadModelParams() {
    const response = await fetch('trained_parameters.json');
    return await response.json();
}

async function createModel() {
    const params = await loadModelParams();
    const l1_weights = tf.tensor2d(params.l1_weights);
    const l1_biases = tf.tensor1d(params.l1_biases);
    const l2_weights = tf.tensor2d(params.l2_weights);
    const l2_biases = tf.tensor1d(params.l2_biases);
    const l3_weights = tf.tensor2d(params.l3_weights);
    const l3_biases = tf.tensor1d(params.l3_biases);
    

    let model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [28*28],
        units: 16,
        activation: 'relu',
    }));

    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
    }));

    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax',
    }));

    model.layers[0].setWeights([l1_weights, l1_biases]);
    model.layers[1].setWeights([l2_weights, l2_biases]);
    model.layers[2].setWeights([l3_weights, l3_biases]);
    console.log(model.layers[0])

    return model;
}

window.onload = () => {
    const canvas = document.getElementById('draw');
    const ctx = canvas.getContext('2d');
    const sideLength = window.innerHeight * 0.4;
    canvas.width = sideLength;
    canvas.height = sideLength;
    let isDrawing = false;
    ctx.strokeStyle = 'white';
    ctx.shadowBlur = 4;
    ctx.shadowColor = '#B7B7B7';
    clear(canvas, ctx);

    const clearbtn = document.getElementById('clear');
    clearbtn.addEventListener('click', () => {
        clear(canvas, ctx);
    });


    const predictbtn = document.getElementById('predict');
    predictbtn.addEventListener('click', () => {
        predict(canvas);
    });

    canvas.addEventListener('mousedown', () => {
        isDrawing = true;
    });

    canvas.addEventListener('mousemove', (e) => {
        if (isDrawing) {
            ctx.lineWidth = sideLength * 0.1;
            ctx.lineCap = 'round';
            ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
            ctx.stroke();
        }
    });

    canvas.addEventListener('mouseup', () => {
        isDrawing = false;
        ctx.beginPath();
    });

    canvas.addEventListener('mouseleave', () => {
        isDrawing = false;
    });
};

function clear(canvas, ctx) {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

async function predict(canvas) {
    model = await createModel();
    const img = new Image();
    img.src = canvas.toDataURL('image/png');
    img.onload = () => {
        const tfImage = tf.browser.fromPixels(img, 1);
        const resized = tf.image.resizeBilinear(tfImage, [28, 28]);
        const input = tf.reshape(resized, [1, 28*28]);
        const prediction = model.predict(input);
        const result = prediction.argMax().dataSync();
    };
}