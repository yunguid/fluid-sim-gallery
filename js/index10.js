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
        maxVelocity: 10, // Increased default maxVelocity for higher responsiveness
        touchForceScale: 1.0, // Increased default touchForceScale for stronger influence
        audioSensitivity: 2.0, // Further increased sensitivity for more pronounced reactions
        noiseScale: 0.005, // New parameter for Perlin noise scale
        dampingFactor: 0.98, // Slightly higher damping for smoother motion
    };

    // Constants for the simulation
    let PARTICLE_DENSITY = PARAMS.particleDensity;
    const MAX_NUM_PARTICLES = 150000; // Increased particle count for denser visuals
    const PARTICLE_LIFETIME = 1000;
    const NUM_JACOBI_STEPS = 5; // Increased steps for more accurate pressure calculations
    const PRESSURE_CALC_ALPHA = -1;
    const PRESSURE_CALC_BETA = 0.25;
    const NUM_RENDER_STEPS = 4; // Increased render steps for smoother particle movement
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
    const audioSmoothingFactor = 0.95; // Increased smoothing for more consistent influence

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
        analyserNode.fftSize = 4096; // Increased FFT size for better frequency resolution
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

        // Compute audio level as the weighted average of frequency data
        const audioLevel = getWeightedAverageVolume(frequencyData) / 255; // Normalize between 0 and 1

        // Smooth the audio level to prevent sudden changes
        smoothedAudioLevel = audioSmoothingFactor * smoothedAudioLevel + (1 - audioSmoothingFactor) * audioLevel;

        // Process frequency data to obtain low, mid, and high frequencies
        frequencyRanges = getFrequencyRanges(frequencyData);

        // Adjust currentMaxVelocity and currentTouchForceScale based on smoothed audio level
        const MIN_MAX_VELOCITY = DEFAULT_MAX_VELOCITY * 0.5; // Lower minimum for greater variability
        const MAX_VELOCITY_BOOST = DEFAULT_MAX_VELOCITY * 4; // Further increased boost
        currentMaxVelocity = MIN_MAX_VELOCITY + smoothedAudioLevel * MAX_VELOCITY_BOOST * PARAMS.audioSensitivity;

        const MIN_TOUCH_FORCE_SCALE = 0.0; // Minimum remains the same
        const MAX_TOUCH_FORCE_SCALE_BOOST = DEFAULT_TOUCH_FORCE_SCALE * 4; // Further increased boost
        currentTouchForceScale = MIN_TOUCH_FORCE_SCALE + smoothedAudioLevel * MAX_TOUCH_FORCE_SCALE_BOOST * PARAMS.audioSensitivity;

        // Limit the maximum values to prevent excessive speeds
        currentMaxVelocity = Math.min(currentMaxVelocity, DEFAULT_MAX_VELOCITY * 10); // Further increased max multiplier
        currentTouchForceScale = Math.min(currentTouchForceScale, DEFAULT_TOUCH_FORCE_SCALE * 10); // Further increased max multiplier

        // Update touch program uniforms
        touch.setUniform('u_touchForceScale', currentTouchForceScale);
        touch.setUniform('u_maxVelocity', currentMaxVelocity);

        // Apply continuous audio-based velocity field
        applyContinuousAudioVelocity(frequencyRanges);

        // Compute target color based on frequency ranges
        targetColor.r = mapRange(frequencyRanges.low, 0, 1, 0.2, 1.0);  // Slightly higher base color
        targetColor.g = mapRange(frequencyRanges.mid, 0, 1, 0.2, 1.0);
        targetColor.b = mapRange(frequencyRanges.high, 0, 1, 0.2, 1.0);

        // Store audioLevel for use in damping
        lastAudioLevel = smoothedAudioLevel;
    }

    // Calculate weighted average volume for smoother influence
    function getWeightedAverageVolume(frequencyData) {
        const bufferLength = frequencyData.length;
        let sum = 0;
        let totalWeight = 0;
        for (let i = 0; i < bufferLength; i++) {
            const weight = i / bufferLength; // Higher frequencies have higher weights
            sum += frequencyData[i] * weight;
            totalWeight += weight;
        }
        return sum / totalWeight;
    }

    // Calculate average volumes for frequency ranges
    function getFrequencyRanges(frequencyData) {
        const bufferLength = frequencyData.length;
        const lowEnd = Math.floor(bufferLength * 0.2); // Increased low frequency range
        const midEnd = Math.floor(bufferLength * 0.6); // Adjusted mid frequency range

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

    // Apply a continuous velocity field influenced by audio frequencies
    function applyContinuousAudioVelocity(frequencyRanges) {
        const { low, mid, high } = frequencyRanges;

        // Generate a base velocity influenced by low frequencies (bass)
        const baseVelocity = [
            mapRange(low, 0, 1, -currentMaxVelocity, currentMaxVelocity),
            mapRange(low, 0, 1, -currentMaxVelocity, currentMaxVelocity),
        ];

        // Add mid frequencies to introduce rotational flows
        const rotationalVelocity = [
            mapRange(mid, 0, 1, -currentMaxVelocity * 0.5, currentMaxVelocity * 0.5),
            mapRange(mid, 0, 1, -currentMaxVelocity * 0.5, currentMaxVelocity * 0.5),
        ];

        // Combine velocities for a cohesive flow
        const combinedVelocity = [
            baseVelocity[0] + rotationalVelocity[0],
            baseVelocity[1] + rotationalVelocity[1],
        ];

        // Introduce Perlin noise for natural turbulence
        const noiseVelocity = [
            perlinNoise(globalTime * PARAMS.noiseScale) * currentMaxVelocity,
            perlinNoise(globalTime * PARAMS.noiseScale + 100) * currentMaxVelocity, // Offset to get different noise
        ];

        // Final velocity by combining audio-driven and noise-driven components
        const finalVelocity = [
            combinedVelocity[0] + noiseVelocity[0],
            combinedVelocity[1] + noiseVelocity[1],
        ];

        // Apply the updated velocity field uniformly using injectVelocity program
        injectVelocity.setUniform('u_vector', finalVelocity);
        composer.step({
            program: injectVelocity,
            input: velocityState,
            output: velocityState,
        });
    }

    // Simple Perlin noise implementation for turbulence
    // Placeholder function: Replace with a proper Perlin noise function or library for better results
    function perlinNoise(x) {
        return (Math.sin(x * 2 * Math.PI) + 1) / 2; // Simple sine wave as a placeholder
    }

    // Global time variable for noise calculation
    let globalTime = 0.0;

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
    let currentColor = { r: 0.2, g: 0.2, b: 1.0 }; // Start with a vibrant blue
    let targetColor = { r: 0.2, g: 0.2, b: 1.0 };
    const COLOR_SMOOTHING = 0.92; // Increased smoothing for gradual color transitions

    // Initialize GPUPrograms for simulation steps

    // Advection program: Moves velocity based on current velocity field
    const advection = new GPUProgram(composer, {
        name: 'advection',
        fragmentShader: `
        in vec2 v_uv;

        uniform sampler2D u_state;
        uniform sampler2D u_velocity;
        uniform vec2 u_dimensions;

        out vec2 out_state;

        void main() {
            // Backtrace to find the source position
            vec2 velocity = texture(u_velocity, v_uv).xy;
            vec2 sourcePos = v_uv - velocity / u_dimensions;
            out_state = texture(u_state, sourcePos).xy;
        }`,
        uniforms: [
            { name: 'u_state', value: 0, type: INT },
            { name: 'u_velocity', value: 1, type: INT },
            { name: 'u_dimensions', value: [canvas.width, canvas.height], type: FLOAT },
        ],
    });

    // Divergence calculation: Measures the divergence of the velocity field
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
            out_divergence = 0.5 * (e - w + n - s);
        }`,
        uniforms: [
            { name: 'u_vectorField', value: 0, type: INT },
            { name: 'u_pxSize', value: [1 / velocityState.width, 1 / velocityState.height], type: FLOAT },
        ],
    });

    // Jacobi iteration: Solves for pressure to enforce incompressibility
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

    // Gradient Subtraction: Removes pressure gradient from velocity to ensure divergence-free field
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

            vec2 gradient = 0.5 * vec2(e - w, n - s);
            vec2 velocity = texture(u_vectorField, v_uv).xy;
            out_result = velocity - gradient;
        }`,
        uniforms: [
            { name: 'u_pxSize', value: [1 / velocityState.width, 1 / velocityState.height], type: FLOAT },
            { name: 'u_scalarField', value: 0, type: INT },
            { name: 'u_vectorField', value: 1, type: INT },
        ],
    });

    // Damping program: Applies damping to velocity to simulate friction
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
            { name: 'u_dampingFactor', value: PARAMS.dampingFactor, type: FLOAT },
        ],
    });

    // RenderParticles program: Determines particle opacity and brightness based on age and velocity
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
            // Smooth fade in and fade out
            float opacity = smoothstep(0.0, FADE_TIME, ageFraction) * smoothstep(1.0 - FADE_TIME, 1.0, ageFraction);
            vec2 velocity = texture(u_velocity, v_uv).xy;
            // Brightness based on velocity magnitude
            float brightness = clamp(length(velocity) * 0.1, 0.0, 1.0);
            out_state = opacity * brightness;
        }`,
        uniforms: [
            { name: 'u_ages', value: 0, type: INT },
            { name: 'u_velocity', value: 1, type: INT },
        ],
    });

    // AgeParticles program: Increments particle age and resets if necessary
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

    // AdvectParticles program: Moves particles based on the velocity field
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
            vec2 velocity = texture(u_velocity, position * pxSize).xy;
            displacement += velocity * 0.5 / u_dimensions; // Half-step for better accuracy

            // Wrap around the screen edges
            absolute = mod(absolute + displacement, u_dimensions);
            displacement = vec2(0.0); // Reset displacement after moving

            // If the particle is too old, reset its position
            int age = texture(u_ages, v_uv).x;
            if (age == 0) {
                absolute = texture(u_initialPositions, v_uv).rg;
            }

            out_position = vec4(absolute, displacement);
        }`,
        uniforms: [
            { name: 'u_positions', value: 0, type: INT },
            { name: 'u_velocity', value: 1, type: INT },
            { name: 'u_ages', value: 2, type: INT },
            { name: 'u_initialPositions', value: 3, type: INT },
            { name: 'u_dimensions', value: [canvas.width, canvas.height], type: FLOAT },
        ],
    });

    // FadeTrails program: Gradually fades the trail texture
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

    // RenderTrails program: Renders the trails with particle colors
    const renderTrails = new GPUProgram(composer, {
        name: 'renderTrails',
        fragmentShader: `
        in vec2 v_uv;
        uniform sampler2D u_trailState;
        uniform vec3 u_particleColor;
        out vec4 out_color;
        void main() {
            vec3 background = vec3(0.0, 0.0, 0.0); // Black background for better contrast
            float trailIntensity = texture(u_trailState, v_uv).x;
            out_color = vec4(mix(background, u_particleColor, pow(trailIntensity, 2.0)), 1);
        }
        `,
        uniforms: [
            { name: 'u_trailState', value: 0, type: INT },
            { name: 'u_particleColor', value: [0.0, 0.5, 1.0], type: FLOAT }, // Default blue color
        ],
    });

    // RenderPressure program: Visualizes the pressure field (optional)
    const renderPressure = renderSignedAmplitudeProgram(composer, {
        name: 'renderPressure',
        type: pressureState.type,
        scale: 0.5,
        component: 'x',
    });

    // Initialize the touch program used to apply continuous velocity
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

    // Initialize the injectVelocity program to uniformly inject velocity vectors
    const injectVelocity = new GPUProgram(composer, {
        name: 'injectVelocity',
        fragmentShader: `
        in vec2 v_uv;

        uniform vec2 u_vector;

        uniform sampler2D u_velocity;

        out vec2 out_velocity;

        void main() {
            // Inject the velocity vector uniformly
            vec2 existingVelocity = texture(u_velocity, v_uv).xy;
            out_velocity = existingVelocity + u_vector;
        }`,
        uniforms: [
            { name: 'u_vector', value: [0, 0], type: FLOAT },
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
        // Update global time for noise calculations
        globalTime += 0.016; // Approx. 60 FPS

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
        damping.setUniform('u_dampingFactor', PARAMS.dampingFactor);
        composer.step({
            program: damping,
            input: velocityState,
            output: velocityState,
        });

        // Apply continuous audio-based velocity field
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
                vectorSpacing: 20, // Increased spacing for clearer vectors
                vectorScale: 3.0, // Increased scale for more visible vectors
                color: [1, 1, 1], // White vectors for better visibility
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
    ui.push(pane.addInput(PARAMS, 'trailLength', { min: 10, max: 100, step: 1, label: 'Trail Length' }).on('change', () => {
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

    ui.push(pane.addInput(PARAMS, 'particleDensity', { min: 0.01, max: 0.2, step: 0.01, label: 'Particle Density' }).on('change', (ev) => {
        // Update PARTICLE_DENSITY and recalculate number of particles
        PARTICLE_DENSITY = ev.value;
        onResize();
    }));

    ui.push(pane.addInput(PARAMS, 'maxVelocity', { min: 5, max: 50, step: 1, label: 'Max Velocity' }).on('change', (ev) => {
        PARAMS.maxVelocity = ev.value;
        DEFAULT_MAX_VELOCITY = PARAMS.maxVelocity;
        currentMaxVelocity = DEFAULT_MAX_VELOCITY;
    }));

    ui.push(pane.addInput(PARAMS, 'touchForceScale', { min: 0.5, max: 10, step: 0.5, label: 'Touch Force Scale' }).on('change', (ev) => {
        PARAMS.touchForceScale = ev.value;
        DEFAULT_TOUCH_FORCE_SCALE = PARAMS.touchForceScale;
        currentTouchForceScale = DEFAULT_TOUCH_FORCE_SCALE;
    }));

    ui.push(pane.addInput(PARAMS, 'audioSensitivity', { min: 1.0, max: 5.0, step: 0.5, label: 'Audio Sensitivity' }).on('change', (ev) => {
        PARAMS.audioSensitivity = ev.value;
    }));

    ui.push(pane.addInput(PARAMS, 'noiseScale', { min: 0.001, max: 0.02, step: 0.001, label: 'Noise Scale' }).on('change', (ev) => {
        PARAMS.noiseScale = ev.value;
    }));

    ui.push(pane.addInput(PARAMS, 'dampingFactor', { min: 0.90, max: 0.99, step: 0.01, label: 'Damping Factor' }).on('change', (ev) => {
        PARAMS.dampingFactor = ev.value;
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
        injectVelocity.dispose();
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
