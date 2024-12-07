'use strict';

{
    const {
        GLSL1,
        GLSL3,
        WEBGL1,
        WEBGL2,
        isWebGL2Supported,
    } = GPUIO;

    // Check if CanvasCapture is available
    const CanvasCapture = window.CanvasCapture ? window.CanvasCapture.CanvasCapture : null;
    const RECORD_FPS = 60;

    // Wait for Tweakpane to be available
    function waitForTweakpane() {
        return new Promise((resolve) => {
            if (window.Tweakpane && window.Tweakpane.Pane) {
                resolve();
            } else {
                window.addEventListener('TweakpaneLoaded', resolve);
            }
        });
    }

    waitForTweakpane().then(() => {
        if (typeof window.Tweakpane === 'undefined' || typeof window.Tweakpane.Pane === 'undefined') {
            console.error('Tweakpane is not defined.');
        } else {
            if (typeof window.main === 'function') { // Check if main is defined
                initializeApplication();
            } else {
                console.error('main() is not defined or not accessible in the global scope');
            }
        }
    });

    function initializeApplication() {
        console.log('Initializing application...');
        // Init a simple gui.
        const pane = new Tweakpane.Pane();

        // Init a pane to toggle main pane visibility.
        const paneToggle = new window.Tweakpane.Pane();
        paneToggle.expanded = false;
        paneToggle.addButton({ title: 'Show Controls' }).on('click', () => {
            paneToggle.expanded = false;
            pane.expanded = true;
        });
        pane.addButton({ title: 'Hide Controls'}).on('click', () => {
            pane.expanded = false;
            paneToggle.expanded = true;
        });

        // Make sure pane is available in the global scope
        window.pane = pane;

        // Init info dialog.
        MicroModal.init();

        // Init an overlay to prevent click events from bubbling through
        // modal to gui or canvas.
        const overlay = document.createElement('div');
        overlay.id = 'touchOverlay';
        overlay.style.width = '100%';
        overlay.style.height = '100%';
        overlay.style.opacity = 0;	
        overlay.style.position = 'absolute';
        overlay.style['z-index'] = 1;
        overlay.style.display = 'none';
        document.body.append(overlay);

        const webGLSettings = {
            webGLVersion: isWebGL2Supported() ? 'WebGL 2' : 'WebGL 1',
            GLSLVersion: isWebGL2Supported() ? 'GLSL 3' : 'GLSL 1',
        };
        const availableWebGLVersions = { webgl1: 'WebGL 1' };
        const availableGLSLVersions = { glsl1: 'GLSL 1' };
        if (isWebGL2Supported()) {
            availableWebGLVersions.webgl2 = 'WebGL 2';
            availableGLSLVersions.glsl3 = 'GLSL 3';
        }

        // Global variables to get from example app.
        let loop, dispose, composer, canvas;
        // Other global ui variables.
        let title = webGLSettings.webGLVersion;
        let useGLSL3Toggle;

        function reloadExampleWithNewParams() {
            if (useGLSL3Toggle) {
                useGLSL3Toggle.dispose();
                useGLSL3Toggle = undefined;
            }
            if (canvas) {
                canvas.removeEventListener('gesturestart', disableZoom);
                canvas.removeEventListener('gesturechange', disableZoom); 
                canvas.removeEventListener('gestureend', disableZoom);
            }
            if (webGLSettings.webGLVersion === 'WebGL 1') webGLSettings.GLSLVersion = 'GLSL 1';
            if (dispose) dispose();
            console.log('Reloading example with new parameters...');
            try {
                ({ loop, composer, dispose, canvas } = window.main({ // Ensure main is called from global scope
                    pane,
                    contextID: webGLSettings.webGLVersion === 'WebGL 2' ? WEBGL2 : WEBGL1,
                    glslVersion: webGLSettings.GLSLVersion === 'GLSL 3' ? GLSL3 : GLSL1,
                }));
            } catch (error) {
                console.error('Error initializing main function:', error);
                return;
            }

            if (!canvas) {
                console.error('Canvas was not returned by main().');
                return;
            }

            canvas.addEventListener('gesturestart', disableZoom);
            canvas.addEventListener('gesturechange', disableZoom); 
            canvas.addEventListener('gestureend', disableZoom);
            
            useGLSL3Toggle = settings.addInput(
                webGLSettings,
                'GLSLVersion',
                {
                    options: webGLSettings.webGLVersion === 'WebGL 2' ? availableGLSLVersions : { glsl1: 'GLSL 1' },
                    label: 'GLSL Version',
                }).on('change', () => {
                    // Some weird issue with calling dispose() inside change callback is throwing error in Tweakpane.
                    // Use timeout to fix it.
                    setTimeout(reloadExampleWithNewParams, 10);
                });
            title = `${webGLSettings.webGLVersion}`;
            settings.title = title;

            if (CanvasCapture) {
                CanvasCapture.init(canvas, { showRecDot: true, showDialogs: true, showAlerts: true, recDotCSS: { left: '0', right: 'auto' } });
                CanvasCapture.bindKeyToVideoRecord('v', {
                    format: CanvasCapture.WEBM,
                    name: 'screen_recording',
                    fps: RECORD_FPS,
                    quality: 1,
                });
            } else {
                console.warn('CanvasCapture is not available. Recording functionality will be disabled.');
            }

        }

        // Add some settings to gui.
        const settings = pane.addFolder({
            title,
            expanded: false,
        });
        settings.addInput(webGLSettings, 'webGLVersion', {
            options: availableWebGLVersions,
            label: 'WebGL Version',
        }).on('change', reloadExampleWithNewParams);

        // Add info modal.
        const modalOptions = {
            showModal: () => {
                // Show/hide overlay, otherwise clicks are passing through due to fixed/abs positioning of modal.
                MicroModal.show('modal-1', { onClose: () => { setTimeout(() => { overlay.style.display = 'none'; }, 500); } });
                overlay.style.display = 'block';
            },
            sourceCode: () => {
                document.getElementById('sourceCode').click();
            },
        }
        pane.addButton({ title: 'About'}).on('click', modalOptions.showModal);
        pane.addButton({ title: 'View Code'}).on('click', modalOptions.sourceCode);

        // Load example app.
        reloadExampleWithNewParams();

        // Disable gestures.
        function disableZoom(e) {
            e.preventDefault();
            const scale = 'scale(1)';
            // @ts-ignore
            document.body.style.webkitTransform =  scale;    // Chrome, Opera, Safari
            // @ts-ignore
            document.body.style.msTransform =   scale;       // IE 9
            // @ts-ignore
            document.body.style.transform = scale;
        }

        let numFrames = 0;

        function outerLoop() {
            // Update fps counter.
            const { fps, numTicks } = composer.tick();
            if (numTicks % 10 === 0) {
                settings.title = `${title} (${fps.toFixed(1)} FPS)`;
            }
            window.requestAnimationFrame(outerLoop);
            // Run example loop.
            if (loop) loop();

            // Screen recording.
            if (CanvasCapture) {
                CanvasCapture.checkHotkeys();
                if (CanvasCapture.isRecording()) {
                    CanvasCapture.recordFrame();
                    numFrames++;
                    console.log(`Recording duration: ${(numFrames / RECORD_FPS).toFixed(2)} sec`);
                } else {
                    numFrames = 0;
                }
            }

            // ... rest of your code ...
        }
        // Start loop.
        outerLoop();
    }
}
