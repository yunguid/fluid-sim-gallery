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
        trailLength: 15,
        render: 'Fluid',
        particleDensity: 0.1,
        maxVelocity: 30,
        touchForceScale: 2,
        sensitivity: {
            low: 1.0,
            mid: 1.0,
            high: 1.0
        },
    };

    // Constants for the simulation
    let TOUCH_FORCE_SCALE = PARAMS.touchForceScale;
    let PARTICLE_DENSITY = PARAMS.particleDensity;
    const MAX_NUM_PARTICLES = 100000;
    let PARTICLE_LIFETIME_BASE = 1000;
    let PARTICLE_LIFETIME = PARTICLE_LIFETIME_BASE;
    const NUM_JACOBI_STEPS = 3;
    const PRESSURE_CALC_ALPHA = -1;
    const PRESSURE_CALC_BETA = 0.25;
    const NUM_RENDER_STEPS = 3;
    const VELOCITY_SCALE_FACTOR = 8;
    let MAX_VELOCITY = PARAMS.maxVelocity;
    const POSITION_NUM_COMPONENTS = 4;
    const FREQUENCY_BANDS = {
        low: { start: 20, end: 250 },      // Bass
        mid: { start: 251, end: 4000 },    // Mids
        high: { start: 4001, end: 20000 }   // Highs
    };
    const FORCE_SMOOTHING = 0.2; // Adjust between 0 (no smoothing) to 1 (instant change)
    const MAX_FORCE = 50; // Upper limit for force magnitudes

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
    let smoothedForces = { low: 0, mid: 0, high: 0 };

    // Get audio elements from the DOM
    const audioInput = document.createElement('input');
    audioInput.type = 'file';
    audioInput.accept = 'audio/*';
    audioInput.id = 'audio-upload';
    audioInput.style.display = 'none';
    document.body.appendChild(audioInput);

    // Handle audio file selection
    audioInput.addEventListener('change', handleAudioUpload);

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
        if (!audioReady || !isPlaying) return;

        analyserNode.getByteFrequencyData(frequencyData);

        // Process frequency data to obtain low, mid, and high frequencies
        frequencyRanges = getFrequencyRanges(frequencyData, audioContext);

        // Apply audio forces to the fluid simulation
        applyAudioForces(frequencyRanges, canvas.width, canvas.height);

        // Update particle colors based on frequencies
        updateColor(frequencyRanges);

        // Adjust particle lifetimes based on low frequencies
        adjustParticleLifetime(frequencyRanges);
    }

    // Calculate average volumes for frequency ranges
    function getFrequencyRanges(frequencyData, audioContext) {
        const sampleRate = audioContext.sampleRate;
        const bufferLength = frequencyData.length;
        const frequencies = new Array(bufferLength).fill(0).map((_, i) => i * sampleRate / 2 / bufferLength);

        const bandData = {
            low: [],
            mid: [],
            high: []
        };

        frequencies.forEach((freq, i) => {
            if (freq >= FREQUENCY_BANDS.low.start && freq <= FREQUENCY_BANDS.low.end) {
                bandData.low.push(frequencyData[i]);
            } else if (freq > FREQUENCY_BANDS.mid.start && freq <= FREQUENCY_BANDS.mid.end) {
                bandData.mid.push(frequencyData[i]);
            } else if (freq > FREQUENCY_BANDS.high.start && freq <= FREQUENCY_BANDS.high.end) {
                bandData.high.push(frequencyData[i]);
            }
        });

        return {
            low: getAverageVolume(bandData.low),
            mid: getAverageVolume(bandData.mid),
            high: getAverageVolume(bandData.high),
        };
    }

    // Helper function to calculate average volume
    function getAverageVolume(array) {
        if (array.length === 0) return 0;
        const sum = array.reduce((a, b) => a + b, 0);
        return sum / array.length / 255; // Normalize to [0,1]
    }

    // Mapping function to scale values
    function mapRange(value, inMin, inMax, outMin, outMax) {
        return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
    }

    // Apply forces to the fluid simulation based on audio frequencies
    function applyAudioForces(frequencyRanges, canvasWidth, canvasHeight) {
        const { low, mid, high } = frequencyRanges;

        // Smooth the forces
        smoothedForces.low += (low - smoothedForces.low) * FORCE_SMOOTHING;
        smoothedForces.mid += (mid - smoothedForces.mid) * FORCE_SMOOTHING;
        smoothedForces.high += (high - smoothedForces.high) * FORCE_SMOOTHING;

        // Map frequency amplitudes to force magnitudes with scaling factors and sensitivity
        const lowForce = Math.min(mapRange(smoothedForces.low, 0, 1, 0, MAX_VELOCITY) * 2 * PARAMS.sensitivity.low, MAX_FORCE);
        const midForce = Math.min(mapRange(smoothedForces.mid, 0, 1, 0, MAX_VELOCITY / 2) * PARAMS.sensitivity.mid, MAX_FORCE);
        const highForce = Math.min(mapRange(smoothedForces.high, 0, 1, 0, MAX_VELOCITY / 3) * PARAMS.sensitivity.high, MAX_FORCE);

        // Define screen regions
        const leftRegion = { x: 0, y: 0, width: canvasWidth / 3, height: canvasHeight };
        const centerRegion = { x: canvasWidth / 3, y: 0, width: canvasWidth / 3, height: canvasHeight };
        const rightRegion = { x: (2 * canvasWidth) / 3, y: 0, width: canvasWidth / 3, height: canvasHeight };

        // Apply forces to each region
        applyForceToRegion(leftRegion, lowForce, 'low');
        applyForceToRegion(centerRegion, midForce, 'mid');
        applyForceToRegion(rightRegion, highForce, 'high');
    }

    function applyForceToRegion(region, forceMagnitude, frequencyType) {
        if (forceMagnitude === 0) return; // No force to apply

        // Determine direction based on frequency type
        let angle;
        switch (frequencyType) {
            case 'low':
                angle = Math.PI / 2; // Downwards
                break;
            case 'mid':
                angle = 0; // Rightwards
                break;
            case 'high':
                angle = -Math.PI / 2; // Upwards
                break;
            default:
                angle = 0;
        }

        const vector = [
            forceMagnitude * Math.cos(angle),
            forceMagnitude * Math.sin(angle),
        ];

        // Set uniforms for the touch program
        touch.setUniform('u_vector', vector);
        touch.setUniform('u_touchForceScale', TOUCH_FORCE_SCALE);
        touch.setUniform('u_maxVelocity', MAX_VELOCITY);

        // Apply the force within the specified region
        composer.stepSegment({
            program: touch,
            input: velocityState,
            output: velocityState,
            position1: [region.x, region.y],
            position2: [region.x + region.width, region.y + region.height],
            thickness: Math.max(region.width, region.height),
            endCaps: false,
        });
    }

    // Calculate the number of particles based on canvas size
    function calcNumParticles(width, height) {
        return Math.min(Math.ceil(width * height * PARTICLE_DENSITY), MAX_NUM_PARTICLES);
    }
    let NUM_PARTICLES = calcNumParticles(canvas.width, canvas.clientHeight);

    // Initialize GPUComposer
    const composer = new GPUComposer({ canvas, contextID, glslVersion });
    console.log('GPUComposer initialized');

    // Initialize GPULayers for simulation state
    const simulationWidth = canvas.clientWidth;
    const simulationHeight = canvas.clientHeight;
    const velocityState = new GPULayer(composer, {
        name: 'velocity',
        dimensions: [Math.ceil(simulationWidth / VELOCITY_SCALE_FACTOR), Math.ceil(simulationHeight / VELOCITY_SCALE_FACTOR)],
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
            {
                name: 'u_state',
                value: 0,
                type: INT,
            },
            {
                name: 'u_velocity',
                value: 1,
                type: INT,
            },
            {
                name: 'u_dimensions',
                value: [canvas.width, canvas.height],
                type: FLOAT,
            },
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
            {
                name: 'u_vectorField',
                value: 0,
                type: INT,
            },
            {
                name: 'u_pxSize',
                value: [1 / velocityState.width, 1 / velocityState.height],
                type: FLOAT,
            }
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

        out vec4 out_jacobi;

        void main() {
            vec4 n = texture(u_previousState, v_uv + vec2(0, u_pxSize.y));
            vec4 s = texture(u_previousState, v_uv - vec2(0, u_pxSize.y));
            vec4 e = texture(u_previousState, v_uv + vec2(u_pxSize.x, 0));
            vec4 w = texture(u_previousState, v_uv - vec2(u_pxSize.x, 0));
            vec4 d = texture(u_divergence, v_uv);
            out_jacobi = (n + s + e + w + u_alpha * d) * u_beta;
        }`,
        uniforms: [
            {
                name: 'u_alpha',
                value: PRESSURE_CALC_ALPHA,
                type: FLOAT,
            },
            {
                name: 'u_beta',
                value: PRESSURE_CALC_BETA,
                type: FLOAT,
            },
            {
                name: 'u_pxSize',
                value: [1 / velocityState.width, 1 / velocityState.height],
                type: FLOAT,
            },
            {
                name: 'u_previousState',
                value: 0,
                type: INT,
            },
            {
                name: 'u_divergence',
                value: 1,
                type: INT,
            },
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
            {
                name: 'u_pxSize',
                value: [1 / velocityState.width, 1 / velocityState.height],
                type: FLOAT,
            },
            {
                name: 'u_scalarField',
                value: 0,
                type: INT,
            },
            {
                name: 'u_vectorField',
                value: 1,
                type: INT,
            },
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
            float opacity = mix(0.0, 1.0, min(ageFraction * 10.0, 1.0)) * mix(1.0, 0.0, max(ageFraction * 10.0 - 90.0, 0.0));
            vec2 velocity = texture(u_velocity, v_uv).xy;
            // Show the fastest regions with darker color.
            float multiplier = clamp(dot(velocity, velocity) * 0.05 + 0.7, 0.0, 1.0);
            out_state = opacity * multiplier;
        }`,
        uniforms: [
            {
                name: 'u_ages',
                value: 0,
                type: INT,
            },
            {
                name: 'u_velocity',
                value: 1,
                type: INT,
            },
        ],
    });
    const ageParticles = new GPUProgram(composer, {
        name: 'ageParticles',
        fragmentShader: `
        in vec2 v_uv;

        uniform isampler2D u_ages;
        uniform float u_lifetime;

        out int out_age;

        void main() {
            int age = texture(u_ages, v_uv).x + 1;
            out_age = stepi(age, int(u_lifetime)) * age;
        }`,
        uniforms: [
            {
                name: 'u_ages',
                value: 0,
                type: INT,
            },
            {
                name: 'u_lifetime',
                value: PARTICLE_LIFETIME,
                type: FLOAT,
            },
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
            // Store small displacements as separate number until they accumulate sufficiently.
            // This prevents small offsets on large abs positions from being lost in float16 precision.
            vec4 positionData = texture(u_positions, v_uv);
            vec2 absolute = positionData.rg;
            vec2 displacement = positionData.ba;
            vec2 position = absolute + displacement;

            // Forward integrate via RK2.
            vec2 pxSize = 1.0 / u_dimensions;
            vec2 velocity1 = texture(u_velocity, position * pxSize).xy;
            vec2 halfStep = position + velocity1 * 0.5 * ${1 / NUM_RENDER_STEPS};
            vec2 velocity2 = texture(u_velocity, halfStep * pxSize).xy;
            displacement += velocity2 * ${1 / NUM_RENDER_STEPS};

            // Merge displacement with absolute if needed.
            float shouldMerge = step(20.0, dot(displacement, displacement));
            // Also wrap absolute position if needed.
            absolute = mod(absolute + shouldMerge * displacement + u_dimensions, u_dimensions);
            displacement *= (1.0 - shouldMerge);

            // If this particle is being reset, give it a random position.
            int shouldReset = stepi(texture(u_ages, v_uv).x, 1);
            out_position = mix(vec4(absolute, displacement), texture(u_initialPositions, v_uv), float(shouldReset));
        }`,
        uniforms: [
            {
                name: 'u_positions',
                value: 0,
                type: INT,
            },
            {
                name: 'u_velocity',
                value: 1,
                type: INT,
            },
            {
                name: 'u_ages',
                value: 2,
                type: INT,
            },
            {
                name: 'u_initialPositions',
                value: 3,
                type: INT,
            },
            {
                name: 'u_dimensions',
                value: [canvas.width, canvas.height],
                type: FLOAT,
            },
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
            {
                name: 'u_image',
                value: 0,
                type: INT,
            },
            {
                name: 'u_increment',
                value: -1 / PARAMS.trailLength,
                type: FLOAT,
            },
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
            // Use u_particleColor for dynamic modulation
            out_color = vec4(mix(background, u_particleColor, pow(texture(u_trailState, v_uv).x, 2.0)), 1);
        }
        `,
        uniforms: [
            {
                name: 'u_particleColor',
                value: [0.0, 0.0, 1.0], // Initial color (blue)
                type: FLOAT, // Adjust if necessary (e.g., FLOAT_VEC3)
            },
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
        in vec2 v_uv_local;

        uniform sampler2D u_velocity;
        uniform vec2 u_vector;
        uniform float u_touchForceScale;
        uniform float u_maxVelocity;

        out vec2 out_velocity;

        void main() {
            vec2 radialVec = (v_uv_local * 2.0 - 1.0);
            float radiusSq = dot(radialVec, radialVec);
            vec2 velocity = texture(u_velocity, v_uv).xy + (1.0 - radiusSq) * u_vector * u_touchForceScale;
            float velocityMag = length(velocity);
            out_velocity = velocity / velocityMag * min(velocityMag, u_maxVelocity);
        }`,
        uniforms: [
            {
                name: 'u_velocity',
                value: 0,
                type: INT,
            },
            {
                name: 'u_vector',
                value: [0, 0],
                type: FLOAT,
            },
            {
                name: 'u_touchForceScale',
                value: TOUCH_FORCE_SCALE,
                type: FLOAT,
            },
            {
                name: 'u_maxVelocity',
                value: MAX_VELOCITY,
                type: FLOAT,
            },
        ],
    });

    // Color modulation variables
    let currentColor = { r: 0.0, g: 0.0, b: 1.0 }; // Start with blue
    let targetColor = { r: 0.0, g: 0.0, b: 1.0 };
    const COLOR_SMOOTHING = 0.8; // Adjust between 0 (no smoothing) to 1 (instant change)

    // Initialize the sensitivity folder in the UI
    const sensitivityFolder = pane.addFolder({ title: 'Frequency Sensitivity' });
    sensitivityFolder.addInput(PARAMS.sensitivity, 'low', { min: 0.1, max: 3.0, step: 0.1, label: 'Low Sensitivity' }).on('change', (ev) => {
        // Sensitivity is applied in force mapping directly
    });
    sensitivityFolder.addInput(PARAMS.sensitivity, 'mid', { min: 0.1, max: 3.0, step: 0.1, label: 'Mid Sensitivity' }).on('change', (ev) => {
        // Sensitivity is applied in force mapping directly
    });
    sensitivityFolder.addInput(PARAMS.sensitivity, 'high', { min: 0.1, max: 3.0, step: 0.1, label: 'High Sensitivity' }).on('change', (ev) => {
        // Sensitivity is applied in force mapping directly
    });

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

        if (isPlaying) {
            visualizeAudio();
            updateColor(frequencyRanges);
        }

        // Smoothly interpolate currentColor towards targetColor
        currentColor.r += (targetColor.r - currentColor.r) * COLOR_SMOOTHING;
        currentColor.g += (targetColor.g - currentColor.g) * COLOR_SMOOTHING;
        currentColor.b += (targetColor.b - currentColor.b) * COLOR_SMOOTHING;

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
        PARTICLE_DENSITY = ev.value;
        onResize(); // Recalculate number of particles
    }));

    ui.push(pane.addInput(PARAMS, 'maxVelocity', { min: 10, max: 100, step: 1, label: 'Max Velocity' }).on('change', (ev) => {
        MAX_VELOCITY = ev.value;
        touch.setUniform('u_maxVelocity', MAX_VELOCITY);
    }));

    ui.push(pane.addInput(PARAMS, 'touchForceScale', { min: 0.5, max: 5, step: 0.1, label: 'Touch Force Scale' }).on('change', (ev) => {
        TOUCH_FORCE_SCALE = ev.value;
        touch.setUniform('u_touchForceScale', TOUCH_FORCE_SCALE);
    }));

    ui.push(pane.addButton({ title: 'Reset' }).on('click', onResize));
    ui.push(pane.addButton({ title: 'Save PNG (p)' }).on('click', savePNG));

    // Add audio controls to Tweakpane
    const audioFolder = pane.addFolder({ title: 'Audio Controls' });

    audioFolder.addButton({ title: 'Upload Audio' }).on('click', () => {
        audioInput.click();
    });

    const playPauseBtn = audioFolder.addButton({ title: 'Play' });
    playPauseBtn.disabled = true;

    playPauseBtn.on('click', togglePlayPause);

    function togglePlayPause() {
        if (!audioReady) return;

        if (isPlaying) {
            audioElement.pause();
            playPauseBtn.title = 'Play';
            playPauseBtn.textContent = 'Play';
        } else {
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
            audioElement.play();
            playPauseBtn.title = 'Pause';
            playPauseBtn.textContent = 'Pause';
        }
        isPlaying = !isPlaying;
    }

    // Function to handle audio upload
    function handleAudioUpload(event) {
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

    // Function to adjust particle lifetime based on low frequencies
    function adjustParticleLifetime(frequencyRanges) {
        const { low } = frequencyRanges;
        PARTICLE_LIFETIME = PARTICLE_LIFETIME_BASE * (1 + low); // Longer lifetimes for stronger low frequencies
        ageParticles.setUniform('u_lifetime', PARTICLE_LIFETIME);
    }

    // Function to update color based on frequency ranges
    function updateColor(frequencyRanges) {
        const { low, mid, high } = frequencyRanges;

        // Define base colors for each frequency band
        const lowColor = { r: 0.0, g: 0.0, b: 1.0 };    // Blue
        const midColor = { r: 0.0, g: 1.0, b: 0.0 };    // Green
        const highColor = { r: 1.0, g: 0.0, b: 0.0 };   // Red

        // Blend colors based on frequency intensities
        targetColor.r = lowColor.r * low + midColor.r * mid + highColor.r * high;
        targetColor.g = lowColor.g * low + midColor.g * mid + highColor.g * high;
        targetColor.b = lowColor.b * low + midColor.b * mid + highColor.b * high;
    }

    // Handle window resize with debouncing
    let resizeTimeout;
    window.addEventListener('resize', onResize);
    function onResize() {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            const width = window.innerWidth;
            const height = window.innerHeight;

            // Resize canvas
            canvas.width = width;
            canvas.height = height;

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
            for (let i = 0; i < NUM_PARTICLES; i++) {
                positions[POSITION_NUM_COMPONENTS * i] = Math.random() * width;
                positions[POSITION_NUM_COMPONENTS * i + 1] = Math.random() * height;
                positions[POSITION_NUM_COMPONENTS * i + 2] = 0.0; // Initial displacement x
                positions[POSITION_NUM_COMPONENTS * i + 3] = 0.0; // Initial displacement y
            }
            particlePositionState.resize(NUM_PARTICLES, positions);
            particleInitialState.resize(NUM_PARTICLES, positions);

            const ages = new Int16Array(NUM_PARTICLES);
            for (let i = 0; i < NUM_PARTICLES; i++) {
                ages[i] = Math.round(Math.random() * PARTICLE_LIFETIME);
            }
            particleAgeState.resize(NUM_PARTICLES, ages);
        }, 200); // Adjust the delay as needed
    }
    onResize();

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
