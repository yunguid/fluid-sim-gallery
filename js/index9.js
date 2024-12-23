// index.js

// Main is called from ../common/wrapper.js
function main({ pane, contextID, glslVersion }) {
    const {
        GPUComposer,
        GPUProgram,
        GPULayer,
        SHORT,
        INT,
        FLOAT,
        REPEAT,
        NEAREST,
        LINEAR,
        renderSignedAmplitudeProgram,
    } = GPUIO;

    // Simulation parameters
    const PARAMS = {
        trailLength: 50,
        render: 'Fluid',
        particleDensity: 0.08,
        maxVelocity: 4,
        touchForceScale: 0.2,
        audioSensitivity: 1.5, // Increased default sensitivity from 0.5 to 1.5
    };

    // Constants for the simulation
    let PARTICLE_DENSITY = PARAMS.particleDensity;
    const MAX_NUM_PARTICLES = 100000;
    const PARTICLE_LIFETIME = 1000;
    const NUM_JACOBI_STEPS = 3;
    const PRESSURE_CALC_ALPHA = -1;
    const PRESSURE_CALC_BETA = 0.25;
    const NUM_RENDER_STEPS = 3;
    const VELOCITY_SCALE_FACTOR = 8;
    const POSITION_NUM_COMPONENTS = 4;

    // Default values for maxVelocity and touchForceScale
    let DEFAULT_MAX_VELOCITY = PARAMS.maxVelocity;
    let DEFAULT_TOUCH_FORCE_SCALE = PARAMS.touchForceScale;

    // Current values that will be adjusted based on audio volume
    let currentMaxVelocity = DEFAULT_MAX_VELOCITY;
    let currentTouchForceScale = DEFAULT_TOUCH_FORCE_SCALE;

    let shouldSavePNG = false;

    // Create and append canvas to the document body
    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas);

    // Audio variables
    let audioContext, analyserNode, sourceNode, audioElement;
    let frequencyData;
    let audioReady = false;
    let frequencyRanges = { low: 0, mid: 0, high: 0 };
    let isPlaying = false;
    let lastAudioLevel = 0.0;
    let smoothedAudioLevel = 0.0;
    const audioSmoothingFactor = 0.9; // Adjust between 0 (no smoothing) to 1 (instant change)

    // Get audio elements from the DOM
    const audioInput = document.getElementById('audio-upload');

    // Handle audio file selection
    audioInput.addEventListener('change', function (event) {
        const file = event.target.files[0];
        if (file) {
            console.log('Audio file selected:', file.name);
            const fileURL = URL.createObjectURL(file);
            audioElement = new Audio();
            audioElement.src = fileURL;
            audioElement.crossOrigin = "anonymous";
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            sourceNode = audioContext.createMediaElementSource(audioElement);

            setupAudioProcessing();
            playPauseBtn.disabled = false;
        }
    });

    // Setup audio processing
    function setupAudioProcessing() {
        console.log('Setting up audio processing');
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 2048; // Adjust for frequency resolution
        sourceNode.connect(analyserNode);
        analyserNode.connect(audioContext.destination);

        const bufferLength = analyserNode.frequencyBinCount;
        frequencyData = new Uint8Array(bufferLength);

        audioReady = true;
    }

    // Visualize audio by processing frequency data
    function visualizeAudio() {
        if (!audioReady || !isPlaying) {
            lastAudioLevel = 0.0;
            return;
        }

        analyserNode.getByteFrequencyData(frequencyData);

        // Compute audio level as the average of the frequency data
        const audioLevel = getAverageVolume(frequencyData) / 255; // Normalize between 0 and 1

        // Smooth the audio level to prevent sudden changes
        smoothedAudioLevel = audioSmoothingFactor * smoothedAudioLevel + (1 - audioSmoothingFactor) * audioLevel;

        // Process frequency data to obtain low, mid, and high frequencies
        frequencyRanges = getFrequencyRanges(frequencyData);

        // Adjust currentMaxVelocity and currentTouchForceScale based on smoothed audio level
        const MIN_MAX_VELOCITY = 0.1; // Minimum value for maxVelocity
        const MAX_VELOCITY_BOOST = DEFAULT_MAX_VELOCITY * 3; // Increased boost amount from 2 to 3
        currentMaxVelocity = MIN_MAX_VELOCITY + smoothedAudioLevel * MAX_VELOCITY_BOOST * PARAMS.audioSensitivity;

        const MIN_TOUCH_FORCE_SCALE = 0.0; // Minimum value for touchForceScale
        const MAX_TOUCH_FORCE_SCALE_BOOST = DEFAULT_TOUCH_FORCE_SCALE * 3; // Increased boost amount from 2 to 3
        currentTouchForceScale = MIN_TOUCH_FORCE_SCALE + smoothedAudioLevel * MAX_TOUCH_FORCE_SCALE_BOOST * PARAMS.audioSensitivity;

        // Limit the maximum values to prevent excessive speeds
        currentMaxVelocity = Math.min(currentMaxVelocity, DEFAULT_MAX_VELOCITY * 5); // Increased max multiplier
        currentTouchForceScale = Math.min(currentTouchForceScale, DEFAULT_TOUCH_FORCE_SCALE * 5); // Increased max multiplier

        // Update touch program uniforms
        touch.setUniform('u_touchForceScale', currentTouchForceScale);
        touch.setUniform('u_maxVelocity', currentMaxVelocity);

        // Apply audio forces to the fluid simulation
        applyAudioForces(frequencyRanges);

        // Compute target color based on frequency ranges
        targetColor.r = mapRange(frequencyRanges.low, 0, 1, 0.0, 1.0);  // Red channel
        targetColor.g = mapRange(frequencyRanges.mid, 0, 1, 0.0, 1.0);  // Green channel
        targetColor.b = mapRange(frequencyRanges.high, 0, 1, 0.0, 1.0); // Blue channel

        // Store audioLevel for use in damping
        lastAudioLevel = smoothedAudioLevel;
    }

    // Calculate average volumes for frequency ranges
    function getFrequencyRanges(frequencyData) {
        const bufferLength = frequencyData.length;
        const lowEnd = Math.floor(bufferLength * 0.1);
        const midEnd = Math.floor(bufferLength * 0.5);

        const lowFreqRange = frequencyData.slice(0, lowEnd);
        const midFreqRange = frequencyData.slice(lowEnd, midEnd);
        const highFreqRange = frequencyData.slice(midEnd);

        const lowFreqAvg = getAverageVolume(lowFreqRange);
        const midFreqAvg = getAverageVolume(midFreqRange);
        const highFreqAvg = getAverageVolume(highFreqRange);

        // Apply logarithmic scaling for better visual representation
        const scaledLow = Math.log1p(lowFreqAvg) / Math.log1p(255);
        const scaledMid = Math.log1p(midFreqAvg) / Math.log1p(255);
        const scaledHigh = Math.log1p(highFreqAvg) / Math.log1p(255);

        return {
            low: scaledLow,
            mid: scaledMid,
            high: scaledHigh,
        };
    }

    // Helper function to calculate average volume
    function getAverageVolume(array) {
        const sum = array.reduce((a, b) => a + b, 0);
        return sum / array.length;
    }

    // Mapping function to scale values
    function mapRange(value, inMin, inMax, outMin, outMax) {
        return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
    }

    // Apply forces to the fluid simulation based on audio frequencies
    function applyAudioForces(frequencyRanges) {
        const { low, mid, high } = frequencyRanges;

        // Map frequency amplitudes to force magnitudes
        const lowForce = mapRange(low, 0, 1, 0, currentMaxVelocity * 2.0); // Increased multiplier from 1.0 to 2.0 for Bass
        const midForce = mapRange(mid, 0, 1, 0, currentMaxVelocity * 1.0); // Increased multiplier from 0.5 to 1.0 for Mid frequencies
        const highForce = mapRange(high, 0, 1, 0, currentMaxVelocity * 0.5); // Increased multiplier from 0.25 to 0.5 for High frequencies

        // Calculate total force
        const totalForce = lowForce + midForce + highForce;

        // Apply multiple larger forces at random positions
        const numForces = 10; // Increased number of forces from 5 to 10
        for (let i = 0; i < numForces; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const vector = [
                totalForce * Math.cos(angle),
                totalForce * Math.sin(angle),
            ];

            const position = [
                Math.random() * canvas.clientWidth,
                Math.random() * canvas.clientHeight,
            ];

            // Apply the force at the random position
            touch.setUniform('u_vector', vector);

            composer.stepCircle({
                program: touch,
                input: velocityState,
                output: velocityState,
                position,
                diameter: 100,
            });
        }
    }

    // Calculate the number of particles based on canvas size
    function calcNumParticles(width, height) {
        return Math.min(Math.ceil(width * height * PARTICLE_DENSITY), MAX_NUM_PARTICLES);
    }

    // Initialize GPUComposer
    const composer = new GPUComposer({ canvas, contextID, glslVersion });
    console.log('GPUComposer initialized');

    // Initialize GPULayers for simulation state
    let NUM_PARTICLES = calcNumParticles(canvas.width, canvas.height);
    const velocityState = new GPULayer(composer, {
        name: 'velocity',
        dimensions: [Math.ceil(canvas.width / VELOCITY_SCALE_FACTOR), Math.ceil(canvas.height / VELOCITY_SCALE_FACTOR)],
        type: FLOAT,
        filter: LINEAR,
        numComponents: 2,
        wrapX: REPEAT,
        wrapY: REPEAT,
        numBuffers: 2,
    });
    const divergenceState = new GPULayer(composer, {
        name: 'divergence',
        dimensions: [velocityState.width, velocityState.height],
        type: FLOAT,
        filter: NEAREST,
        numComponents: 1,
        wrapX: REPEAT,
        wrapY: REPEAT,
    });
    const pressureState = new GPULayer(composer, {
        name: 'pressure',
        dimensions: [velocityState.width, velocityState.height],
        type: FLOAT,
        filter: NEAREST,
        numComponents: 1,
        wrapX: REPEAT,
        wrapY: REPEAT,
        numBuffers: 2,
    });
    const particlePositionState = new GPULayer(composer, {
        name: 'position',
        dimensions: NUM_PARTICLES,
        type: FLOAT,
        numComponents: POSITION_NUM_COMPONENTS,
        numBuffers: 2,
    });
    const particleInitialState = new GPULayer(composer, {
        name: 'initialPosition',
        dimensions: NUM_PARTICLES,
        type: FLOAT,
        numComponents: POSITION_NUM_COMPONENTS,
        numBuffers: 1,
    });
    const particleAgeState = new GPULayer(composer, {
        name: 'age',
        dimensions: NUM_PARTICLES,
        type: SHORT,
        numComponents: 1,
        numBuffers: 2,
    });
    const trailState = new GPULayer(composer, {
        name: 'trails',
        dimensions: [canvas.width, canvas.height],
        type: FLOAT,
        filter: NEAREST,
        numComponents: 1,
        numBuffers: 2,
    });

    // Color modulation variables
    let currentColor = { r: 0.0, g: 0.0, b: 1.0 }; // Start with blue
    let targetColor = { r: 0.0, g: 0.0, b: 1.0 };
    const COLOR_SMOOTHING = 0.9; // Adjust between 0 (no smoothing) to 1 (instant change)

    // Initialize GPUPrograms for simulation steps
    const advection = new GPUProgram(composer, {
        name: 'advection',
        fragmentShader: `
        in vec2 v_uv;

        uniform sampler2D u_state;
        uniform sampler2D u_velocity;
        uniform vec2 u_dimensions;

        out vec2 out_state;

        void main() {
            // Implicitly solve advection.
            out_state = texture(u_state, v_uv - texture(u_velocity, v_uv).xy / u_dimensions).xy;
        }`,
        uniforms: [
            { name: 'u_state', value: 0, type: INT },
            { name: 'u_velocity', value: 1, type: INT },
            { name: 'u_dimensions', value: [canvas.width, canvas.height], type: FLOAT },
        ],
    });

    const divergence2D = new GPUProgram(composer, {
        name: 'divergence2D',
        fragmentShader: `
        in vec2 v_uv;

        uniform sampler2D u_vectorField;
        uniform vec2 u_pxSize;

        out float out_divergence;

        void main() {
            float n = texture(u_vectorField, v_uv + vec2(0, u_pxSize.y)).y;
            float s = texture(u_vectorField, v_uv - vec2(0, u_pxSize.y)).y;
            float e = texture(u_vectorField, v_uv + vec2(u_pxSize.x, 0)).x;
            float w = texture(u_vectorField, v_uv - vec2(u_pxSize.x, 0)).x;
            out_divergence = 0.5 * ( e - w + n - s);
        }`,
        uniforms: [
            { name: 'u_vectorField', value: 0, type: INT },
            { name: 'u_pxSize', value: [1 / velocityState.width, 1 / velocityState.height], type: FLOAT },
        ],
    });

    const jacobi = new GPUProgram(composer, {
        name: 'jacobi',
        fragmentShader: `
        in vec2 v_uv;

        uniform float u_alpha;
        uniform float u_beta;
        uniform vec2 u_pxSize;
        uniform sampler2D u_previousState;
        uniform sampler2D u_divergence;

        out float out_jacobi;

        void main() {
            float n = texture(u_previousState, v_uv + vec2(0, u_pxSize.y)).r;
            float s = texture(u_previousState, v_uv - vec2(0, u_pxSize.y)).r;
            float e = texture(u_previousState, v_uv + vec2(u_pxSize.x, 0)).r;
            float w = texture(u_previousState, v_uv - vec2(u_pxSize.x, 0)).r;
            float d = texture(u_divergence, v_uv).r;
            out_jacobi = (n + s + e + w + u_alpha * d) * u_beta;
        }`,
        uniforms: [
            { name: 'u_alpha', value: PRESSURE_CALC_ALPHA, type: FLOAT },
            { name: 'u_beta', value: PRESSURE_CALC_BETA, type: FLOAT },
            { name: 'u_pxSize', value: [1 / velocityState.width, 1 / velocityState.height], type: FLOAT },
            { name: 'u_previousState', value: 0, type: INT },
            { name: 'u_divergence', value: 1, type: INT },
        ],
    });

    const gradientSubtraction = new GPUProgram(composer, {
        name: 'gradientSubtraction',
        fragmentShader: `
        in vec2 v_uv;

        uniform vec2 u_pxSize;
        uniform sampler2D u_scalarField;
        uniform sampler2D u_vectorField;

        out vec2 out_result;

        void main() {
            float n = texture(u_scalarField, v_uv + vec2(0, u_pxSize.y)).r;
            float s = texture(u_scalarField, v_uv - vec2(0, u_pxSize.y)).r;
            float e = texture(u_scalarField, v_uv + vec2(u_pxSize.x, 0)).r;
            float w = texture(u_scalarField, v_uv - vec2(u_pxSize.x, 0)).r;

            out_result = texture(u_vectorField, v_uv).xy - 0.5 * vec2(e - w, n - s);
        }`,
        uniforms: [
            { name: 'u_pxSize', value: [1 / velocityState.width, 1 / velocityState.height], type: FLOAT },
            { name: 'u_scalarField', value: 0, type: INT },
            { name: 'u_vectorField', value: 1, type: INT },
        ],
    });

    const damping = new GPUProgram(composer, {
        name: 'damping',
        fragmentShader: `
        in vec2 v_uv;

        uniform sampler2D u_velocity;
        uniform float u_dampingFactor;

        out vec2 out_velocity;

        void main() {
            vec2 velocity = texture(u_velocity, v_uv).xy;
            out_velocity = velocity * u_dampingFactor;
        }`,
        uniforms: [
            { name: 'u_velocity', value: 0, type: INT },
            { name: 'u_dampingFactor', value: 1.0, type: FLOAT },
        ],
    });

    const renderParticles = new GPUProgram(composer, {
        name: 'renderParticles',
        fragmentShader: `
        #define FADE_TIME 0.1

        in vec2 v_uv;
        in vec2 v_uv_position;

        uniform isampler2D u_ages;
        uniform sampler2D u_velocity;

        out float out_state;

        void main() {
            float ageFraction = float(texture(u_ages, v_uv_position).x) / ${PARTICLE_LIFETIME.toFixed(1)};
            // Fade first 10% and last 10%.
            float opacity = mix(0.0, 1.0, min(ageFraction * 10.0, 1.0)) * mix(1.0, 0.0, max(ageFraction * 10.0 - 9.0, 0.0));
            vec2 velocity = texture(u_velocity, v_uv).xy;
            // Show the fastest regions with darker color.
            float multiplier = clamp(length(velocity) * 0.05 + 0.7, 0.0, 1.0);
            out_state = opacity * multiplier;
        }`,
        uniforms: [
            { name: 'u_ages', value: 0, type: INT },
            { name: 'u_velocity', value: 1, type: INT },
        ],
    });

    const ageParticles = new GPUProgram(composer, {
        name: 'ageParticles',
        fragmentShader: `
        in vec2 v_uv;

        uniform isampler2D u_ages;

        out int out_age;

        void main() {
            int age = texture(u_ages, v_uv).x + 1;
            out_age = (age <= ${PARTICLE_LIFETIME}) ? age : 0;
        }`,
        uniforms: [
            { name: 'u_ages', value: 0, type: INT },
        ],
    });

    const advectParticles = new GPUProgram(composer, {
        name: 'advectParticles',
        fragmentShader: `
        in vec2 v_uv;

        uniform vec2 u_dimensions;
        uniform sampler2D u_positions;
        uniform sampler2D u_velocity;
        uniform isampler2D u_ages;
        uniform sampler2D u_initialPositions;

        out vec4 out_position;

        void main() {
            vec4 positionData = texture(u_positions, v_uv);
            vec2 absolute = positionData.rg;
            vec2 displacement = positionData.ba;
            vec2 position = absolute + displacement;

            vec2 pxSize = 1.0 / u_dimensions;
            vec2 velocity1 = texture(u_velocity, position * pxSize).xy;
            vec2 halfStep = position + velocity1 * 0.5 * ${1 / NUM_RENDER_STEPS};
            vec2 velocity2 = texture(u_velocity, halfStep * pxSize).xy;
            displacement += velocity2 * ${1 / NUM_RENDER_STEPS};

            // Merge displacement with absolute if needed.
            float shouldMerge = step(20.0, dot(displacement, displacement));
            absolute = mod(absolute + shouldMerge * displacement + u_dimensions, u_dimensions);
            displacement *= (1.0 - shouldMerge);

            // If this particle is being reset, give it a random position.
            int shouldReset = stepi(texture(u_ages, v_uv).x, 1);
            out_position = mix(vec4(absolute, displacement), texture(u_initialPositions, v_uv), float(shouldReset));
        }`,
        uniforms: [
            { name: 'u_positions', value: 0, type: INT },
            { name: 'u_velocity', value: 1, type: INT },
            { name: 'u_ages', value: 2, type: INT },
            { name: 'u_initialPositions', value: 3, type: INT },
            { name: 'u_dimensions', value: [canvas.width, canvas.height], type: FLOAT },
        ],
    });

    const fadeTrails = new GPUProgram(composer, {
        name: 'fadeTrails',
        fragmentShader: `
        in vec2 v_uv;

        uniform sampler2D u_image;
        uniform float u_increment;

        out float out_color;

        void main() {
            out_color = max(texture(u_image, v_uv).x + u_increment, 0.0);
        }`,
        uniforms: [
            { name: 'u_image', value: 0, type: INT },
            { name: 'u_increment', value: -1 / PARAMS.trailLength, type: FLOAT },
        ],
    });

    const renderTrails = new GPUProgram(composer, {
        name: 'renderTrails',
        fragmentShader: `
        in vec2 v_uv;
        uniform sampler2D u_trailState;
        uniform vec3 u_particleColor;
        out vec4 out_color;
        void main() {
            vec3 background = vec3(0.1, 0.1, 0.2); // Darker blue background
            out_color = vec4(mix(background, u_particleColor, pow(texture(u_trailState, v_uv).x, 2.0)), 1);
        }
        `,
        uniforms: [
            { name: 'u_trailState', value: 0, type: INT },
            { name: 'u_particleColor', value: [0.0, 0.0, 1.0], type: FLOAT },
        ],
    });

    const renderPressure = renderSignedAmplitudeProgram(composer, {
        name: 'renderPressure',
        type: pressureState.type,
        scale: 0.5,
        component: 'x',
    });

    // Initialize the touch program used to apply forces
    const touch = new GPUProgram(composer, {
        name: 'touch',
        fragmentShader: `
        in vec2 v_uv;

        uniform sampler2D u_velocity;
        uniform vec2 u_vector;
        uniform float u_touchForceScale;
        uniform float u_maxVelocity;

        out vec2 out_velocity;

        void main() {
            vec2 velocity = texture(u_velocity, v_uv).xy + u_vector * u_touchForceScale;
            float velocityMag = length(velocity);
            out_velocity = velocity / max(velocityMag, 1e-6) * min(velocityMag, u_maxVelocity);
        }`,
        uniforms: [
            { name: 'u_velocity', value: 0, type: INT },
            { name: 'u_vector', value: [0, 0], type: FLOAT },
            { name: 'u_touchForceScale', value: currentTouchForceScale, type: FLOAT },
            { name: 'u_maxVelocity', value: currentMaxVelocity, type: FLOAT },
        ],
    });

    // Handle window resize
    window.addEventListener('resize', onResize);
    function onResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;

        // Resize composer
        composer.resize([width, height]);

        // Re-initialize textures at new size
        const velocityDimensions = [Math.ceil(width / VELOCITY_SCALE_FACTOR), Math.ceil(height / VELOCITY_SCALE_FACTOR)];
        velocityState.resize(velocityDimensions);
        divergenceState.resize(velocityDimensions);
        pressureState.resize(velocityDimensions);
        trailState.resize([width, height]);

        // Update uniforms
        advection.setUniform('u_dimensions', [width, height]);
        advectParticles.setUniform('u_dimensions', [width, height]);
        const velocityPxSize = [1 / velocityDimensions[0], 1 / velocityDimensions[1]];
        divergence2D.setUniform('u_pxSize', velocityPxSize);
        jacobi.setUniform('u_pxSize', velocityPxSize);
        gradientSubtraction.setUniform('u_pxSize', velocityPxSize);

        // Re-initialize particles
        NUM_PARTICLES = calcNumParticles(width, height);
        const positions = new Float32Array(NUM_PARTICLES * 4);
        for (let i = 0; i < positions.length / 4; i++) {
            positions[POSITION_NUM_COMPONENTS * i] = Math.random() * width;
            positions[POSITION_NUM_COMPONENTS * i + 1] = Math.random() * height;
            positions[POSITION_NUM_COMPONENTS * i + 2] = 0;
            positions[POSITION_NUM_COMPONENTS * i + 3] = 0;
        }
        particlePositionState.resize(NUM_PARTICLES, positions);
        particleInitialState.resize(NUM_PARTICLES, positions);

        const ages = new Int16Array(NUM_PARTICLES);
        for (let i = 0; i < NUM_PARTICLES; i++) {
            ages[i] = Math.round(Math.random() * PARTICLE_LIFETIME);
        }
        particleAgeState.resize(NUM_PARTICLES, ages);
    }
    onResize();

    // Main simulation loop
    function loop() {
        // Advect the velocity vector field
        composer.step({
            program: advection,
            input: [velocityState, velocityState],
            output: velocityState,
        });
        // Compute divergence of advected velocity field
        composer.step({
            program: divergence2D,
            input: velocityState,
            output: divergenceState,
        });
        // Compute the pressure gradient of the advected velocity vector field
        for (let i = 0; i < NUM_JACOBI_STEPS; i++) {
            composer.step({
                program: jacobi,
                input: [pressureState, divergenceState],
                output: pressureState,
            });
        }
        // Subtract the pressure gradient from velocity to obtain a velocity vector field with zero divergence
        composer.step({
            program: gradientSubtraction,
            input: [pressureState, velocityState],
            output: velocityState,
        });

        // Apply damping to the velocity field based on audio level
        const minDamping = 0.7; // Reduced minimum damping to allow higher velocities
        const dampingRange = 1.0 - minDamping;
        const dampingFactor = minDamping + dampingRange * (1.0 - lastAudioLevel * PARAMS.audioSensitivity);
        damping.setUniform('u_dampingFactor', dampingFactor);
        composer.step({
            program: damping,
            input: velocityState,
            output: velocityState,
        });

        // Apply audio forces and compute target color
        visualizeAudio();

        // Smoothly interpolate currentColor towards targetColor
        currentColor.r += (targetColor.r - currentColor.r) * (1 - COLOR_SMOOTHING);
        currentColor.g += (targetColor.g - currentColor.g) * (1 - COLOR_SMOOTHING);
        currentColor.b += (targetColor.b - currentColor.b) * (1 - COLOR_SMOOTHING);

        // Update the u_particleColor uniform in renderTrails
        renderTrails.setUniform('u_particleColor', [currentColor.r, currentColor.g, currentColor.b]);

        // Render based on selected parameter
        if (PARAMS.render === 'Pressure') {
            composer.step({
                program: renderPressure,
                input: pressureState,
            });
        } else if (PARAMS.render === 'Velocity') {
            composer.drawLayerAsVectorField({
                layer: velocityState,
                vectorSpacing: 10,
                vectorScale: 2.5,
                color: [0, 0, 0],
            });
        } else {
            // Increment particle age
            composer.step({
                program: ageParticles,
                input: particleAgeState,
                output: particleAgeState,
            });
            // Fade current trails
            composer.step({
                program: fadeTrails,
                input: trailState,
                output: trailState,
            });
            // Advect particles and render them
            for (let i = 0; i < NUM_RENDER_STEPS; i++) {
                composer.step({
                    program: advectParticles,
                    input: [particlePositionState, velocityState, particleAgeState, particleInitialState],
                    output: particlePositionState,
                });
                composer.drawLayerAsPoints({
                    layer: particlePositionState,
                    program: renderParticles,
                    input: [particleAgeState, velocityState],
                    output: trailState,
                    wrapX: true,
                    wrapY: true,
                });
            }
            // Render particle trails to screen
            composer.step({
                program: renderTrails,
                input: trailState,
            });
        }

        // Save PNG if requested
        if (shouldSavePNG) {
            composer.savePNG({ filename: `fluid` });
            shouldSavePNG = false;
        }
    }

    // UI controls and event listeners
    const ui = [];
    ui.push(pane.addInput(PARAMS, 'trailLength', { min: 0, max: 100, step: 1, label: 'Trail Length' }).on('change', () => {
        fadeTrails.setUniform('u_increment', -1 / PARAMS.trailLength);
    }));
    ui.push(pane.addInput(PARAMS, 'render', {
        options: {
            Fluid: 'Fluid',
            Pressure: 'Pressure',
            Velocity: 'Velocity',
        },
        label: 'Render',
    }));

    ui.push(pane.addInput(PARAMS, 'particleDensity', { min: 0.01, max: 0.5, step: 0.01, label: 'Particle Density' }).on('change', (ev) => {
        // Update PARTICLE_DENSITY and recalculate number of particles
        PARTICLE_DENSITY = ev.value;
        onResize();
    }));

    ui.push(pane.addInput(PARAMS, 'maxVelocity', { min: 1, max: 50, step: 1, label: 'Max Velocity' }).on('change', (ev) => {
        PARAMS.maxVelocity = ev.value;
        DEFAULT_MAX_VELOCITY = PARAMS.maxVelocity;
        currentMaxVelocity = DEFAULT_MAX_VELOCITY;
    }));

    ui.push(pane.addInput(PARAMS, 'touchForceScale', { min: 0.1, max: 5, step: 0.1, label: 'Touch Force Scale' }).on('change', (ev) => {
        PARAMS.touchForceScale = ev.value;
        DEFAULT_TOUCH_FORCE_SCALE = PARAMS.touchForceScale;
        currentTouchForceScale = DEFAULT_TOUCH_FORCE_SCALE;
    }));

    ui.push(pane.addInput(PARAMS, 'audioSensitivity', { min: 0.1, max: 3.0, step: 0.1, label: 'Audio Sensitivity' }).on('change', (ev) => {
        PARAMS.audioSensitivity = ev.value;
    }));

    ui.push(pane.addButton({ title: 'Reset' }).on('click', onResize));
    ui.push(pane.addButton({ title: 'Save PNG (p)' }).on('click', savePNG));

    // Add audio controls to Tweakpane
    const audioFolder = pane.addFolder({ title: 'Audio Controls' });

    audioFolder.addButton({ title: 'Upload Audio' }).on('click', () => {
        audioInput.click();
    });

    const playPauseBtn = audioFolder.addButton({ title: 'Play' }); // Updated initial title to 'Play'
    playPauseBtn.disabled = true;

    playPauseBtn.on('click', togglePlayPause);

    function togglePlayPause() {
        if (!audioReady) return;

        if (isPlaying) {
            audioElement.pause();
            playPauseBtn.title = 'Play';
        } else {
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
            audioElement.play();
            playPauseBtn.title = 'Pause';
        }
        isPlaying = !isPlaying;
    }

    // Function to save PNG
    function savePNG() {
        shouldSavePNG = true;
    }
    window.addEventListener('keydown', onKeydown);
    function onKeydown(e) {
        if (e.key === 'p') {
            savePNG();
        }
    }

    // Animation loop
    function animate() {
        loop();
        requestAnimationFrame(animate);
    }
    animate();

    // Cleanup function
    function dispose() {
        document.body.removeChild(canvas);
        window.removeEventListener('keydown', onKeydown);
        window.removeEventListener('resize', onResize);
        velocityState.dispose();
        divergenceState.dispose();
        pressureState.dispose();
        particlePositionState.dispose();
        particleInitialState.dispose();
        particleAgeState.dispose();
        trailState.dispose();
        advection.dispose();
        divergence2D.dispose();
        jacobi.dispose();
        gradientSubtraction.dispose();
        damping.dispose();
        renderParticles.dispose();
        ageParticles.dispose();
        advectParticles.dispose();
        renderTrails.dispose();
        fadeTrails.dispose();
        renderPressure.dispose();
        touch.dispose();
        composer.dispose();
        ui.forEach(el => {
            pane.remove(el);
        });
        ui.length = 0;
    }

    // Return the main functions and objects
    return {
        loop,
        dispose,
        composer,
        canvas,
    };
}
