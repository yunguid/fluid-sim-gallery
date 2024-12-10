// index.js

// Add this at the top
console.log('Loading updated version:', new Date().toISOString());

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
    };

    const MAX_NUM_PARTICLES = 100000;
    const PARTICLE_LIFETIME = 1000;
    const NUM_JACOBI_STEPS = 3;
    const PRESSURE_CALC_ALPHA = -1;
    const PRESSURE_CALC_BETA = 0.25;
    const NUM_RENDER_STEPS = 3;
    const VELOCITY_SCALE_FACTOR = 8;
    const POSITION_NUM_COMPONENTS = 4;

    const DEFAULT_MAX_VELOCITY = PARAMS.maxVelocity;
    const DEFAULT_TOUCH_FORCE_SCALE = PARAMS.touchForceScale;

    // Tuning constants
    const TURBULENCE_STRENGTH = 0.03;
    const CENTER_PUSH_FORCE = 1.5; // Force applied from center on strong bass hits
    const VELOCITY_DAMPING_FACTOR_NO_MUSIC = 0.99;
    const AMPLITUDE_DROP_THRESHOLD = 0.3; // If amplitude drops by this factor, reduce maxVel quickly
    const AMPLITUDE_DROP_DECAY = 0.8; // How quickly max velocity decays on amplitude drop

    let shouldSavePNG = false;

    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas);

    // Audio variables
    let audioContext, analyserNode, sourceNode, audioElement;
    let frequencyData;
    let audioReady = false;
    let frequencyRanges = { low: 0, mid: 0, high: 0 };
    let isPlaying = false;

    // Amplitude tracking for drops
    let amplitudeEMA = 0;
    const amplitudeSmoothingFactor = 0.95;
    const peakThresholdFactor = 1.5;

    let currentMaxVelocity = PARAMS.maxVelocity;
    let currentTouchForceScale = PARAMS.touchForceScale;
    let currentParticleDensity = PARAMS.particleDensity;
    let currentTrailLength = PARAMS.trailLength;

    let currentColor = { r: 0.0, g: 0.0, b: 1.0 };
    let targetColor = { r: 0.0, g: 0.0, b: 1.0 };
    const COLOR_SMOOTHING = 0.9;
    const valueSmoothingFactor = 0.8;

    let brightness = 1.0;
    let targetBrightness = 1.0;

    let globalTime = 0.0;

    function hueRotate(color, angle) {
        const c = { r: color.r, g: color.g, b: color.b };
        const cosA = Math.cos(angle);
        const sinA = Math.sin(angle);
        const mat = [
            0.213 + 0.787*cosA - 0.213*sinA, 0.715 - 0.715*cosA - 0.715*sinA, 0.072 - 0.072*cosA + 0.928*sinA,
            0.213 - 0.213*cosA + 0.143*sinA, 0.715 + 0.285*cosA + 0.140*sinA, 0.072 - 0.072*cosA - 0.283*sinA,
            0.213 - 0.213*cosA - 0.787*sinA, 0.715 - 0.715*cosA + 0.715*sinA, 0.072 + 0.928*cosA + 0.072*sinA,
        ];
        const R = c.r*mat[0] + c.g*mat[1] + c.b*mat[2];
        const G = c.r*mat[3] + c.g*mat[4] + c.b*mat[5];
        const B = c.r*mat[6] + c.g*mat[7] + c.b*mat[8];
        return { r: R, g: G, b: B };
    }

    const audioInput = document.getElementById('audio-upload');
    audioInput.addEventListener('change', function (event) {
        const file = event.target.files[0];
        if (file) {
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

    function setupAudioProcessing() {
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 2048;
        sourceNode.connect(analyserNode);
        analyserNode.connect(audioContext.destination);

        const bufferLength = analyserNode.frequencyBinCount;
        frequencyData = new Uint8Array(bufferLength);

        audioReady = true;
    }

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

        const scaledLow = Math.log1p(lowFreqAvg) / Math.log1p(255);
        const scaledMid = Math.log1p(midFreqAvg) / Math.log1p(255);
        const scaledHigh = Math.log1p(highFreqAvg) / Math.log1p(255);

        return {
            low: scaledLow,
            mid: scaledMid,
            high: scaledHigh,
        };
    }

    function getAverageVolume(array) {
        const sum = array.reduce((a, b) => a + b, 0);
        return sum / array.length;
    }

    function mapRange(value, inMin, inMax, outMin, outMax) {
        return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
    }

    function visualizeAudio() {
        if (!audioReady || !isPlaying) {
            return;
        }

        analyserNode.getByteFrequencyData(frequencyData);
        frequencyRanges = getFrequencyRanges(frequencyData);
        const maxAmplitude = Math.max(...frequencyData) / 255;

        // Track amplitude EMA
        const prevAmplitudeEMA = amplitudeEMA;
        amplitudeEMA = amplitudeSmoothingFactor * amplitudeEMA + (1 - amplitudeSmoothingFactor) * maxAmplitude;

        // Check if there's a sudden amplitude drop
        if (amplitudeEMA < prevAmplitudeEMA * (1 - AMPLITUDE_DROP_THRESHOLD)) {
            // Sudden drop detected, reduce maxVelocity quickly
            currentMaxVelocity *= AMPLITUDE_DROP_DECAY;
        }

        let targetMaxVelocity = DEFAULT_MAX_VELOCITY;
        let targetTouchForceScale = DEFAULT_TOUCH_FORCE_SCALE;

        const isLoud = maxAmplitude > amplitudeEMA * peakThresholdFactor;
        if (isLoud) {
            // Slightly increase velocity on peaks, but no flashes
            const MAX_VELOCITY_BOOST = 1.5;
            targetMaxVelocity += MAX_VELOCITY_BOOST;
            // No brightness flash, just keep calm
            targetBrightness = 1.0;
        } else {
            targetBrightness = 1.0;
        }

        const { low, mid, high } = frequencyRanges;

        // Scale with frequency ranges but ensure balanced colors and not too much green
        // We'll map low to more bluish tones, mid to warmer tones, and high to add subtle brightness
        // Base color: Blue-ish. Increase R with mid, G with low, B with high but keep it subtle.
        // We'll actually mix frequencies into a balanced color:
        // r ~ mid, g ~ low, b ~ high, each capped to avoid dominance
        const rVal = Math.min(1.0, mid * 0.8 + 0.2);
        const gVal = Math.min(1.0, low * 0.8 + 0.2);
        const bVal = Math.min(1.0, high * 0.8 + 0.2);
        targetColor = { r: rVal, g: gVal, b: bVal };

        // Adjust velocity based on frequencies
        const lowScaledMaxVel = mapRange(low, 0, 1, 1, 10);
        const lowScaledTouchForce = mapRange(low, 0, 1, 0.1, 1.2);
        targetMaxVelocity = Math.max(targetMaxVelocity, lowScaledMaxVel);
        targetTouchForceScale = Math.max(targetTouchForceScale, lowScaledTouchForce);

        const targetParticleDensity = mapRange(mid, 0, 1, 0.01, 0.3);
        const targetTrailLength = mapRange(high, 0, 1, 0, 80); // shorter trails for less clutter

        currentMaxVelocity += (targetMaxVelocity - currentMaxVelocity) * (1 - valueSmoothingFactor);
        currentTouchForceScale += (targetTouchForceScale - currentTouchForceScale) * (1 - valueSmoothingFactor);
        currentParticleDensity += (targetParticleDensity - currentParticleDensity) * (1 - valueSmoothingFactor);
        currentTrailLength += (targetTrailLength - currentTrailLength) * (1 - valueSmoothingFactor);

        PARAMS.maxVelocity = currentMaxVelocity;
        PARAMS.touchForceScale = currentTouchForceScale;
        PARAMS.trailLength = currentTrailLength;
        fadeTrails.setUniform('u_increment', -1 / PARAMS.trailLength);

        const densityDiff = Math.abs(PARAMS.particleDensity - currentParticleDensity);
        PARAMS.particleDensity = currentParticleDensity;
        if (densityDiff > 0.02) {
            onResize();
        }

        applyAudioForces(frequencyRanges, maxAmplitude, prevAmplitudeEMA);
    }

    function applyAudioForces(frequencyRanges, maxAmplitude, prevAmplitudeEMA) {
        const { low, mid, high } = frequencyRanges;

        // Bass reaction: On strong bass, apply a big push from the center
        if (low > 0.5) {
            // Strong center force on bass hits
            applyCenterForce(low);
        }

        // Mid frequencies stabilize flow, no special random events needed
        // High frequencies add subtle turbulence
        if (high > 0.3) {
            applyHighFreqTurbulence(high);
        }

        // Apply a few random directional forces based on total force
        const lowForce = mapRange(low, 0, 1, 0, currentMaxVelocity * 0.4);
        const midForce = mapRange(mid, 0, 1, 0, currentMaxVelocity * 0.2);
        const highForce = mapRange(high, 0, 1, 0, currentMaxVelocity * 0.1);
        const totalForce = lowForce + midForce + highForce;

        touch.setUniform('u_touchForceScale', currentTouchForceScale);
        touch.setUniform('u_maxVelocity', currentMaxVelocity);

        const numForces = 2; // fewer random forces to reduce clutter
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
            touch.setUniform('u_vector', vector);
            composer.stepCircle({
                program: touch,
                input: velocityState,
                output: velocityState,
                position,
                diameter: 60,
            });
        }
    }

    function applyCenterForce(low) {
        // Push outward from center on strong bass
        const angleCount = 8;
        for (let i = 0; i < angleCount; i++) {
            const angle = (i / angleCount) * Math.PI * 2;
            const force = CENTER_PUSH_FORCE * currentMaxVelocity * low;
            const vector = [force * Math.cos(angle), force * Math.sin(angle)];
            touch.setUniform('u_vector', vector);
            composer.stepCircle({
                program: touch,
                input: velocityState,
                output: velocityState,
                position: [canvas.clientWidth * 0.5, canvas.clientHeight * 0.5],
                diameter: 120,
            });
        }
    }

    function applyHighFreqTurbulence(high) {
        // Small turbulence spots for high frequencies
        const turbulenceCount = Math.round(high * 3);
        for (let i = 0; i < turbulenceCount; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const force = currentMaxVelocity * 0.1;
            const vector = [force * Math.cos(angle), force * Math.sin(angle)];
            touch.setUniform('u_vector', vector);
            composer.stepCircle({
                program: touch,
                input: velocityState,
                output: velocityState,
                position: [
                    Math.random() * canvas.clientWidth,
                    Math.random() * canvas.clientHeight
                ],
                diameter: 40,
            });
        }
    }

    function calcNumParticles(width, height) {
        return Math.min(Math.ceil(width * height * PARAMS.particleDensity), MAX_NUM_PARTICLES);
    }

    const composer = new GPUComposer({ canvas, contextID, glslVersion });

    let NUM_PARTICLES = 0;

    const velocityState = new GPULayer(composer, {
        name: 'velocity',
        dimensions: [1, 1],
        type: FLOAT,
        filter: LINEAR,
        numComponents: 2,
        wrapX: REPEAT,
        wrapY: REPEAT,
        numBuffers: 2,
    });
    const divergenceState = new GPULayer(composer, {
        name: 'divergence',
        dimensions: [1, 1],
        type: FLOAT,
        filter: NEAREST,
        numComponents: 1,
        wrapX: REPEAT,
        wrapY: REPEAT,
    });
    const pressureState = new GPULayer(composer, {
        name: 'pressure',
        dimensions: [1, 1],
        type: FLOAT,
        filter: NEAREST,
        numComponents: 1,
        wrapX: REPEAT,
        wrapY: REPEAT,
        numBuffers: 2,
    });
    const particlePositionState = new GPULayer(composer, {
        name: 'position',
        dimensions: 1,
        type: FLOAT,
        numComponents: POSITION_NUM_COMPONENTS,
        numBuffers: 2,
    });
    const particleInitialState = new GPULayer(composer, {
        name: 'initialPosition',
        dimensions: 1,
        type: FLOAT,
        numComponents: POSITION_NUM_COMPONENTS,
        numBuffers: 1,
    });
    const particleAgeState = new GPULayer(composer, {
        name: 'age',
        dimensions: 1,
        type: SHORT,
        numComponents: 1,
        numBuffers: 2,
    });
    const trailState = new GPULayer(composer, {
        name: 'trails',
        dimensions: [1, 1],
        type: FLOAT,
        filter: NEAREST,
        numComponents: 1,
        numBuffers: 2,
    });

    const advection = new GPUProgram(composer, {
        name: 'advection',
        fragmentShader: `
        in vec2 v_uv;

        uniform sampler2D u_state;
        uniform sampler2D u_velocity;
        uniform vec2 u_dimensions;

        out vec2 out_state;

        void main() {
            vec2 velocity = texture(u_velocity, v_uv).xy;
            vec2 backUV = v_uv - velocity / u_dimensions;
            out_state = texture(u_state, backUV).xy;
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
            out_divergence = 0.5 * (e - w + n - s);
        }`,
        uniforms: [
            { name: 'u_vectorField', value: 0, type: INT },
            { name: 'u_pxSize', value: [1, 1], type: FLOAT },
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
            float divVal = texture(u_divergence, v_uv).r;
            out_jacobi = (n + s + e + w + u_alpha * divVal) * u_beta;
        }`,
        uniforms: [
            { name: 'u_alpha', value: PRESSURE_CALC_ALPHA, type: FLOAT },
            { name: 'u_beta', value: PRESSURE_CALC_BETA, type: FLOAT },
            { name: 'u_pxSize', value: [1, 1], type: FLOAT },
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
            { name: 'u_pxSize', value: [1, 1], type: FLOAT },
            { name: 'u_scalarField', value: 0, type: INT },
            { name: 'u_vectorField', value: 1, type: INT },
        ],
    });

    const renderParticles = new GPUProgram(composer, {
        name: 'renderParticles',
        fragmentShader: `
        in vec2 v_uv;
        in vec2 v_uv_position;

        uniform isampler2D u_ages;
        uniform sampler2D u_velocity;

        out float out_state;

        void main() {
            int age = texture(u_ages, v_uv_position).x;
            float ageFraction = float(age) / ${PARTICLE_LIFETIME.toFixed(1)};
            float opacity = min(ageFraction * 10.0, 1.0) * mix(1.0, 0.0, max(ageFraction * 10.0 - 9.0, 0.0));
            vec2 velocity = texture(u_velocity, v_uv).xy;
            float speed = dot(velocity, velocity);
            float multiplier = clamp(speed * 0.02 + 0.7, 0.0, 1.0);
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

            float distSqr = dot(displacement, displacement);
            float mergeThreshold = 20.0;
            float shouldMerge = (distSqr > mergeThreshold) ? 1.0 : 0.0;

            absolute = mod(absolute + shouldMerge * displacement + u_dimensions, u_dimensions);
            displacement *= (1.0 - shouldMerge);

            int ageVal = texture(u_ages, v_uv).x;
            int shouldReset = (ageVal == 1) ? 1 : 0;
            vec4 initialPos = texture(u_initialPositions, v_uv);

            out_position = mix(vec4(absolute, displacement), initialPos, float(shouldReset));
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
            vec3 background = vec3(0.05, 0.05, 0.1);
            float t = texture(u_trailState, v_uv).x;
            out_color = vec4(mix(background, u_particleColor, t * t), 1);
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

    const turbulenceField = new GPUProgram(composer, {
        name: 'turbulenceField',
        fragmentShader: `
        in vec2 v_uv;
        uniform sampler2D u_velocity;
        uniform vec2 u_dimensions;
        uniform float u_strength;
        uniform float u_time;
        
        float rand(vec2 co){
            return fract(sin(dot(co.xy ,vec2(12.9898,78.233)))*43758.5453);
        }

        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            float a = rand(i);
            float b = rand(i + vec2(1.0, 0.0));
            float c = rand(i + vec2(0.0, 1.0));
            float d = rand(i + vec2(1.0, 1.0));
            vec2 u = f*f*(3.0-2.0*f);
            return mix(a, b, u.x) +
                   (c - a)*u.y*(1.0 - u.x) +
                   (d - b)*u.x*u.y;
        }

        out vec2 out_velocity;
        void main() {
            vec2 uv = v_uv * u_dimensions;
            float n = noise(vec2(uv.x * 0.01 + u_time * 0.001, uv.y * 0.01 + u_time * 0.001)) - 0.5;
            vec2 velocity = texture(u_velocity, v_uv).xy;
            velocity += normalize(vec2(cos(n*10.0), sin(n*10.0))) * u_strength;
            out_velocity = velocity;
        }`,
        uniforms: [
            { name: 'u_velocity', value: 0, type: INT },
            { name: 'u_dimensions', value: [1, 1], type: FLOAT },
            { name: 'u_strength', value: TURBULENCE_STRENGTH, type: FLOAT },
            { name: 'u_time', value: 0, type: FLOAT }
        ],
    });

    const velocityDamping = new GPUProgram(composer, {
        name: 'velocityDamping',
        fragmentShader: `
        in vec2 v_uv;
        uniform sampler2D u_velocity;
        uniform float u_damping;

        out vec2 out_velocity;

        void main() {
            vec2 vel = texture(u_velocity, v_uv).xy;
            out_velocity = vel * u_damping;
        }`,
        uniforms: [
            { name: 'u_velocity', value: 0, type: INT },
            { name: 'u_damping', value: VELOCITY_DAMPING_FACTOR_NO_MUSIC, type: FLOAT },
        ],
    });

    window.addEventListener('resize', onResize);
    function onResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;

        composer.resize([width, height]);

        const velocityDimensions = [Math.ceil(width / VELOCITY_SCALE_FACTOR), Math.ceil(height / VELOCITY_SCALE_FACTOR)];
        velocityState.resize(velocityDimensions);
        divergenceState.resize(velocityDimensions);
        pressureState.resize(velocityDimensions);
        trailState.resize([width, height]);

        advection.setUniform('u_dimensions', [width, height]);
        advectParticles.setUniform('u_dimensions', [width, height]);
        const velocityPxSize = [1 / velocityDimensions[0], 1 / velocityDimensions[1]];
        divergence2D.setUniform('u_pxSize', velocityPxSize);
        jacobi.setUniform('u_pxSize', velocityPxSize);
        gradientSubtraction.setUniform('u_pxSize', velocityPxSize);

        NUM_PARTICLES = calcNumParticles(width, height);
        const positions = new Float32Array(NUM_PARTICLES * POSITION_NUM_COMPONENTS);
        for (let i = 0; i < NUM_PARTICLES; i++) {
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

    function loop() {
        composer.step({
            program: advection,
            input: [velocityState, velocityState],
            output: velocityState,
        });

        composer.step({
            program: divergence2D,
            input: velocityState,
            output: divergenceState,
        });

        for (let i = 0; i < NUM_JACOBI_STEPS; i++) {
            composer.step({
                program: jacobi,
                input: [pressureState, divergenceState],
                output: pressureState,
            });
        }

        composer.step({
            program: gradientSubtraction,
            input: [pressureState, velocityState],
            output: velocityState,
        });

        visualizeAudio();

        // Update colors smoothly
        currentColor.r += (targetColor.r - currentColor.r) * (1 - COLOR_SMOOTHING);
        currentColor.g += (targetColor.g - currentColor.g) * (1 - COLOR_SMOOTHING);
        currentColor.b += (targetColor.b - currentColor.b) * (1 - COLOR_SMOOTHING);
        brightness += (targetBrightness - brightness) * 0.05;

        const hueAngle = globalTime * 0.00005; 
        const rotatedColor = hueRotate(currentColor, hueAngle);
        const adjustedColor = [
            Math.min(1.0, rotatedColor.r * brightness),
            Math.min(1.0, rotatedColor.g * brightness),
            Math.min(1.0, rotatedColor.b * brightness)
        ];
        renderTrails.setUniform('u_particleColor', adjustedColor);

        turbulenceField.setUniform('u_time', globalTime);
        turbulenceField.setUniform('u_strength', TURBULENCE_STRENGTH);
        turbulenceField.setUniform('u_dimensions', [Math.ceil(canvas.width / VELOCITY_SCALE_FACTOR), Math.ceil(canvas.height / VELOCITY_SCALE_FACTOR)]);
        composer.step({
            program: turbulenceField,
            input: velocityState,
            output: velocityState,
        });

        // If not playing music, apply damping
        if (!isPlaying) {
            composer.step({
                program: velocityDamping,
                input: velocityState,
                output: velocityState,
            });
        }

        composer.step({
            program: ageParticles,
            input: particleAgeState,
            output: particleAgeState,
        });

        composer.step({
            program: fadeTrails,
            input: trailState,
            output: trailState,
        });

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
            composer.step({
                program: renderTrails,
                input: trailState,
            });
        }

        if (shouldSavePNG) {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            composer.savePNG({ filename: `fluid_${timestamp}` });
            shouldSavePNG = false;
        }
    }

    const ui = [];
    
    // Keep only audio controls
    const audioFolder = pane.addFolder({ title: 'Audio Controls' });
    audioFolder.addButton({ title: 'Upload Audio' }).on('click', () => {
        audioInput.click();
    });

    const playPauseBtn = audioFolder.addButton({ title: 'Play/Pause' });
    playPauseBtn.disabled = true;
    playPauseBtn.on('click', togglePlayPause);

    function togglePlayPause() {
        if (!audioReady) return;
        if (isPlaying) {
            audioElement.pause();
            playPauseBtn.title = 'Play/Pause';
        } else {
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
            audioElement.play();
            playPauseBtn.title = 'Play/Pause';
        }
        isPlaying = !isPlaying;
    }

    function savePNG() {
        shouldSavePNG = true;
    }

    window.addEventListener('keydown', onKeydown);
    function onKeydown(e) {
        if (e.key === 'p') {
            savePNG();
        }
    }

    function animate() {
        globalTime++;
        loop();
        requestAnimationFrame(animate);
    }
    animate();

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
        turbulenceField.dispose();
        velocityDamping.dispose();
        composer.dispose();

        ui.forEach(el => pane.remove(el));
        ui.length = 0;
    }

    return {
        loop,
        dispose,
        composer,
        canvas,
    };
}