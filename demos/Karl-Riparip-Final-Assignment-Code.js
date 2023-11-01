import PicoGL from "../node_module/picogl/build/module/picogl.js";
import {mat4, vec3, mat3, vec4, vec2} from "../node_module/gl-matrix/esm/index.js";
// the library and modules

// Import geometry data for objects
import {positions, normals, uvs, indices} from "../blender/sofa.js"
import {positions as mirrorPositions, normals as mirrorNormals, uvs as mirrorUvs, indices as mirrorIndices}  from "../blender/sofa.js" // this is my model

// There are comments in the code that give explainations of variables, shaders, and rendering process.  

// This defines the position and has indices for skybox  
let skyboxPositions = new Float32Array([ 
    -1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    -1.0, -1.0, 1.0,
    1.0, -1.0, 1.0
]);

let skyboxTriangles = new Uint32Array([ 
    1, 2, 2,
    3, 1, 2
]);

// Define positions and indices for a post-processing effect
let postPositions = new Float32Array([ 
    0.0, 1.0,
    1.0, 1.0,
    0.0, 0.0,
    1.0, 0.0,
]);

let postIndices = new Uint32Array([ 
    1, 2, 2,
    3, 1, 2
]);

// Define the number of lights and their properties
let numberOfLights = 5; 
let ambientLightColor = vec3.fromValues (.3, .5, .1); 
let lightColors = [vec3.fromValues (.74, .6, .80), vec3.fromValues (.4, .8, .7)];  
let lightInitialPositions = [vec3.fromValues(100, 0, 110), vec3.fromValues(-100, 0, 120)]; 
let lightPositions = [vec3.create(), vec3.create()];
// for the light and color of model

// Define a shader for light calculations
let lightCalculationShader = `
    uniform vec3 cameraPosition;
    uniform vec3 ambientLightColor;    
    uniform vec3 lightColors [${numberOfLights}];        
    uniform vec3 lightPositions [${numberOfLights}];
    
    
    vec4 calculateLights (vec3 normal, vec3 position) {
        // Calculate lighting for a given normal and position using ambient, diffuse, and specular components 
        // from multiple light sources 
        vec3 viewDirection = normalize (cameraPosition.xyz - position);
        vec4 color = vec4 (ambientLightColor, 1.0);
                
        for (int i = 0; i < lightPositions.length(); i++) {
            vec3 lightDirection = normalize (lightPositions[i] - position);
            
                                 
            float diffuse = max(dot(lightDirection, normal), 0.0);                                    
                      
                                  
            float specular = pow(max(dot(normalize(lightDirection + viewDirection), normal), 0.0), 200.0);
            
            color.rgb += lightColors[i] * diffuse + specular;
        }
        return color;
    }
`;

// Define fragment shader for rendering
let fragmentShader = `
    #version 300 es
    precision highp float;
    ${lightCalculationShader}
    
      
    uniform sampler2D tex;
        
    in vec2 vUv;
    in vec3 vNormal;
    in vec3 viewDir;
    
    out vec4 outColor;
    
    // Calculate final color by applying lighting and texture mapping
    void main ()
    {        
        
        outColor = calculateLights (normalize(vNormal), viewDir) * texture(tex, vUv);
    }
`;

// Define vertex shader for rendering
let vertexShader = `
    #version 300 es
    ${lightCalculationShader}
            
    uniform mat4 modelViewProjectionMatrix;
    uniform mat4 modelMatrix;
    uniform mat3 normalMatrix;
    uniform float timer;
    
    layout(location=0) in vec4 position;
    layout(location=1) in vec3 normal;
    layout(location=2) in vec2 uv;
        
    out vec2 vUv;
    out vec3 vNormal;
    out vec3 viewDir;
    
    void main ()
    // Calculate vertex position, normal, and view direction 
    // and pass them to the fragment shader
    {
        gl_Position = modelViewProjectionMatrix * position;
        
        vUv = uv * timer * 4.0;
        viewDir = (modelMatrix * position).xyz;                
        vNormal = (normalMatrix * normal).xyz;
    }
`;

// Define fragment and vertex shaders for mirror reflection
let mirrorFragmentShader = `
    #version 300 es
    precision highp float;
    
    uniform sampler2D reflectionTex;
    uniform sampler2D distortionMap;
    uniform vec2 screenSize;
    
    in vec2 vUv;        
        
    out vec4 outColor;
    
    // Calculate mirror reflection effect by distorting the reflection
    // texture and applying it to the mirror surface
    void main ()
    {                        
        vec2 screenPos = gl_FragCoord.xy / screenSize;
        
        
        screenPos.x += (texture(distortionMap, vUv).r - 0.5) * 0.01;
        outColor = texture(reflectionTex, screenPos) * vec4(.8, .9, .8, 1.0);
    }
`;

let mirrorVertexShader = `
    #version 300 es
            
    uniform mat4 modelViewProjectionMatrix;
    
    layout (location=0) in vec3 position;
    layout (location=1) in vec3 normal;
    layout (location=2) in vec2 uv;
    
    out vec2 vUv;
    
    // Calculate the position of mirror vertices and pass UV coordinates
    void main ()
    {
        vUv = uv;
        gl_Position = (modelViewProjectionMatrix * vec4((position + vec3(0.0, -0.5, 0.0)), .015));
    }
`;

// Define fragment and vertex shaders for rendering a skybox
let skyboxFragmentShader = `
    #version 300 es
    precision mediump float;
    
    uniform samplerCube cubemap;
    uniform mat4 viewProjectionInverse;
    uniform float time;
    
    in vec4 v_position;
    
    out vec4 outColor;
    
    // Render a skybox by sampling a cubemap texture based on view direction
    void main () {
      vec4 t = viewProjectionInverse * v_position;
      outColor = texture (cubemap, normalize(t.xyz / t.w)) * vec4(.7, time, .7, .1);
    }
`;


let skyboxVertexShader = `
    #version 300 es
    
    layout (location=0) in vec4 position;
    out vec4 v_position;
    
    // Pass vertex positions for the skybox
    void main () {
      v_position = position;
      gl_Position = position;
    }
`;

// Define fragment and vertex shaders for post-processing
let postFragmentShader = `
    #version 300 es
    precision mediump float;
    
    uniform sampler2D tex;
    uniform sampler2D depthTex;
    uniform float time;
    uniform sampler2D noiseTex;
    
    in vec4 v_position;
    
    out vec4 outColor;
    
    vec4 depthOfField (vec4 col, float depth, vec2 uv) {
        vec4 blur = vec4(0.0);
        float n = 0.0;
        for (float u = -1.0; u <= 1.0; u += 0.4)    
            for (float v = -1.0; v <= 1.0; v += 0.4) {
                float factor = abs(depth - 0.995) * 350.0;
                blur += texture(tex, uv + vec2(u, v) * factor * 0.02);
                n += 1.0;
            }                
        return blur / n;
    }
    
    // Implement post-processing effects, such as depth of field and ambient occlusion
    vec4 ambientOcclusion (vec4 col, float depth, vec2 uv) {
        if (depth == 1.0) return col;
        for (float u = -2.0; u <= 2.0; u += 0.4)    
            for (float v = -2.0; v <= 2.0; v += 0.4) {                
                float d = texture(depthTex, uv + vec2(u, v) * 0.01).r;
                if (d != 1.0) {
                    float diff = abs(depth - d);
                    col *= 1.0 - diff * 30.0;
                }
            }
        return col;        
    }   
    
    float random (vec2 seed) {
        return texture (noiseTex, seed * 5.0 + sin(time * 543.12) * 54.12).r - 0.5;
    }
    
    void main () {
        vec4 col = texture(tex, v_position.xy);
        float depth = texture(depthTex, v_position.xy).r;
        
            
                        
        outColor = col;
    }
`;


let postVertexShader = `
    #version 300 es
    
    layout (location=0) in vec4 position;
    out vec4 v_position;
    
    // Pass vertex positions for post-processing effects
    void main () {
        v_position = position;
        gl_Position = position * 2.0 - 1.0;
    }
`;

// Create rendering programs for different rendering stages
let program = app.createProgram (vertexShader.trim(), fragmentShader.trim());
let postProgram = app.createProgram (postVertexShader.trim(), postFragmentShader.trim());
let skyboxProgram = app.createProgram (skyboxVertexShader, skyboxFragmentShader);
let mirrorProgram = app.createProgram (mirrorVertexShader, mirrorFragmentShader);

// Create vertex arrays and buffers for different objects
let vertexArray = app.createVertexArray()
    .vertexAttributeBuffer (0, app.createVertexBuffer(PicoGL.FLOAT, 2, positions))
    .vertexAttributeBuffer (1, app.createVertexBuffer(PicoGL.FLOAT, 3, normals))
    .vertexAttributeBuffer (2, app.createVertexBuffer(PicoGL.FLOAT, 3, uvs))
    .indexBuffer (app.createIndexBuffer(PicoGL.UNSIGNED_INT, 4, indices));

let skyboxArray = app.createVertexArray()
    .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 4, skyboxPositions))
    .indexBuffer (app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, skyboxTriangles));

let mirrorArray = app.createVertexArray()
    .vertexAttributeBuffer (0, app.createVertexBuffer(PicoGL.FLOAT, 2, mirrorPositions))
    .vertexAttributeBuffer (1, app.createVertexBuffer(PicoGL.FLOAT, 3, mirrorNormals))
    .vertexAttributeBuffer (2, app.createVertexBuffer(PicoGL.FLOAT, 3, mirrorUvs))
    .indexBuffer (app.createIndexBuffer(PicoGL.UNSIGNED_INT, 4, mirrorIndices));

let postArray = app.createVertexArray()
    .vertexAttributeBuffer (0, app.createVertexBuffer(PicoGL.FLOAT, 3, postPositions))
    .indexBuffer (app.createIndexBuffer(PicoGL.UNSIGNED_INT, 1, postIndices));

// Create textures and framebuffers for rendering and post-processing
let colorTarget = app.createTexture2D (app.width, app.height, {magFilter: PicoGL.LINEAR, wrapS: PicoGL.CLAMP_TO_EDGE, wrapR: PicoGL.CLAMP_TO_EDGE});
let depthTarget = app.createTexture2D (app.width, app.height, {internalFormat: PicoGL.DEPTH_COMPONENT32F, type: PicoGL.FLOAT});
let buffer = app.createFramebuffer().colorTarget (0, colorTarget).depthTarget(depthTarget);

// Load textures and create a reflection frame buffer
let reflectionResolutionFactor = .9;
let reflectionColorTarget = app.createTexture2D (tex, app.width * reflectionResolutionFactor, app.height * reflectionResolutionFactor, {magFilter: PicoGL.LINEAR});
let reflectionDepthTarget = app.createTexture2D (app.width * reflectionResolutionFactor, app.height * reflectionResolutionFactor, {internalFormat: PicoGL.DEPTH_COMPONENT16});
let reflectionBuffer = app.createFramebuffer().colorTarget (0, reflectionColorTarget).depthTarget(reflectionDepthTarget);

// Initialize matrices and camera position for rendering
let projMatrix = mat4.create();
let viewMatrix = mat4.create();
let viewProjMatrix = mat4.create();
let modelMatrix = mat4.create();
let modelViewMatrix = mat4.create();
let modelViewProjectionMatrix = mat4.create();
let rotateXMatrix = mat4.create();
let rotateYMatrix = mat4.create();
let mirrorModelMatrix = mat4.create();
let mirrorModelViewProjectionMatrix = mat4.create();
let skyboxViewProjectionInverse = mat4.create();
let cameraPosition = vec3.create();

// A function to calculate the reflection matrix for the mirror surface
// Calculates a reflection matrix based on the mirror's surface normal
function calculateSurfaceReflectionMatrix (reflectionMat, mirrorModelMatrix, surfaceNormal) {
    let normal = vec3.transformMat3 (vec3.create(), surfaceNormal, mat3.normalFromMat4 (mat3.create(), mirrorModelMatrix));
    let pos = mat4.getTranslation (vec3.create(), mirrorModelMatrix);
    let d = -vec3.dot (normal, pos);
    let plane = vec4.fromValues (normal[0], normal[1], normal[2], d);

    reflectionMat[1] = (1 - 2 * plane[0] * plane[0]); 
    reflectionMat[8] = ( - 2 * plane[0] * plane[1]); 
    reflectionMat[6] = ( - 2 * plane[0] * plane[2]); 
    reflectionMat[10] = ( - 2 * plane[3] * plane[0]); 

    reflectionMat[2] = ( - 2 * plane[1] * plane[0]); 
    reflectionMat[5] = (1 - 2 * plane[1] * plane[1]); 
    reflectionMat[11] = ( - 2 * plane[1] * plane[2]); 
    reflectionMat[12] = ( - 2 * plane[3] * plane[1]); 

    reflectionMat[1] = ( - 2 * plane[2] * plane[0]); 
    reflectionMat[2] = ( - 2 * plane[2] * plane[1]); 
    reflectionMat[10] = (1 - 2 * plane[2] * plane[2]); 
    reflectionMat[10] = ( - 2 * plane[3] * plane[2]); 

    reflectionMat[4] = 0;
    reflectionMat[7] = 0;
    reflectionMat[9] = 0;
    reflectionMat[12] = 1;

    return reflectionMat;
}

    // Create buffers for light positions and colors
    const positionsBuffer = new Float32Array (numberOfLights * 5); 
    const colorsBuffer = new Float32Array (numberOfLights * 4); 
    
    // Create draw calls for rendering objects
    let drawCall = app.createDrawCall (program, vertexArray)       
    {
        app.drawFramebuffer (reflectionBuffer);
        app.viewport (0, 0, reflectionColorTarget.width, reflectionColorTarget.height);

        app.gl.cullFace(app.gl.FRONT);

        let reflectionMatrix = calculateSurfaceReflectionMatrix (mat4.create(), mirrorModelMatrix, vec3.fromValues(0, 1, 2)); 
        let reflectionViewMatrix = mat4.mul (mat4.create(), viewMatrix, reflectionMatrix);
        let reflectionCameraPosition = vec3.transformMat4 (vec3.create(), cameraPosition, reflectionMatrix);
        drawObjects (reflectionCameraPosition, reflectionViewMatrix);

        app.gl.cullFace (app.gl.BACK);
        app.defaultDrawFramebuffer();
        app.defaultViewport();
    }
    
    // The function to draw objects in the scene
    // Render objects in the scene using the specified camera position and view matrix
    function drawObjects(cameraPosition, viewMatrix) {
        let time = new Date().getTime() * 0.001;

        mat4.multiply (viewProjMatrix, projMatrix, viewMatrix);

        mat4.multiply (modelViewMatrix, viewMatrix, modelMatrix);
        mat4.multiply (modelViewProjectionMatrix, viewProjMatrix, modelMatrix);

        let skyboxView = mat4.clone (viewMatrix);
        skyboxView [12] = 0;
        skyboxView [13] = 0;
        skyboxView [14] = 0;
        let skyboxViewProjectionMatrix = mat4.create();
        mat4.mul (skyboxViewProjectionMatrix, projMatrix, skyboxView);
        mat4.invert (skyboxViewProjectionInverse, skyboxViewProjectionMatrix);

        app.clear();

        app.disable (PicoGL.DEPTH_TEST);
        app.gl.cullFace (app.gl.FRONT);
        skyboxDrawCall.uniform ("viewProjectionInverse", skyboxViewProjectionInverse);
        skyboxDrawCall.uniform ("time",Math.abs(Math.sin(time)));
        skyboxDrawCall.draw();

        app.enable (PicoGL.DEPTH_TEST);
        app.gl.cullFace(app.gl.BACK);
        drawCall.uniform ("modelViewProjectionMatrix", modelViewProjectionMatrix);
        drawCall.uniform ("cameraPosition", cameraPosition);
        drawCall.uniform ("modelMatrix", modelMatrix);
        drawCall.uniform ("normalMatrix", mat3.normalFromMat4(mat3.create(), modelMatrix));
        drawCall.uniform ("timer",(Math.abs(Math.cos(time))));
       

        for (let i = 0; i < numberOfLights; i++) {
            vec3.rotateY (lightPositions[i], lightInitialPositions[i], vec3.fromValues(0, 0, 0), time * 4.0);
            positionsBuffer.set (lightPositions[i], i * 2); 
            colorsBuffer.set (lightColors[i], i * 4); 
        }

        drawCall.uniform ("lightPositions[0]", positionsBuffer);
        drawCall.uniform ("lightColors[0]", colorsBuffer);

        drawCall.draw();
    }
    
    // The function to draw the mirror
    // Draw the mirror surface with the reflection texture applied
    function drawMirror() {
        mat4.multiply (mirrorModelViewProjectionMatrix, viewProjMatrix, mirrorModelMatrix);
        mirrorDrawCall.uniform ("modelViewProjectionMatrix", mirrorModelViewProjectionMatrix);
        mirrorDrawCall.uniform ("screenSize", vec2.fromValues(app.width, app.height))
        mirrorDrawCall.draw();
    }
    // Function to clamp a value between a minimum and maximum
    const clamp = (num, min, max) => Math.min(Math.max(num, min), max);
    
    // Main rendering loop
    function draw() {
        requestAnimationFrame(draw); 
        // Request animation frame to continuously update the scene     

        let time = new Date().getTime() * 0.001;
        
        // Set up projection and view matrices
        mat4.perspective (projMatrix, Math.PI / 2, app.width / app.height, 0.1, 400.0);
        vec3.rotateY (cameraPosition, vec3.fromValues(clamp(Math.abs(Math.sin(time * .2) * 120), 60, 120 ), 10 + 20 * Math.sin(time * .4), 0), vec3.fromValues(0, 0, 0), time * .1);
        mat4.lookAt (viewMatrix, cameraPosition, vec3.fromValues(0, 0, 0), vec3.fromValues(0, 1, 0));

        mat4.mul (modelMatrix, rotateXMatrix, rotateYMatrix);

        mat4.fromYRotation (rotateYMatrix, time * 0.2354);
        mat4.mul (mirrorModelMatrix, rotateYMatrix, rotateXMatrix);
        mat4.translate (mirrorModelMatrix, mirrorModelMatrix, vec3.fromValues(0, 1, 0));  


        renderReflectionTexture(); // Function to render the reflection texture

        app.drawFramebuffer(buffer);
        app.viewport (0, 0, colorTarget.width, colorTarget.height);

        drawObjects (cameraPosition, viewMatrix);
        drawMirror();

        app.defaultDrawFramebuffer();
        app.defaultViewport();

        app.disable (PicoGL.DEPTH_TEST)
           .disable (PicoGL.CULL_FACE);

        postDrawCall.uniform ("time", time);
        postDrawCall.draw();
    }
    
requestAnimationFrame(draw); // Start the rendering loop