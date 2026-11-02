// 3D preview of the canvas in a frame, with Three.js. Sizes are in inches, as the real object.
// The frame is only a preview: it is not part of the built files.

(function () {
    "use strict";

    const VENDOR = "/assets/vendor/three/";
    const WOOD = "/assets/vendor/wood/";
    // width in inches on a 20 in canvas, depth of the moulding as a share of its width
    const FRAMES = {
        oak: {kind: "wood", texture: "Wood049", base: "#b88d57", width: 1.5, profile: "flat", roughness: 0.62},
        pine: {kind: "wood", texture: "Wood049", base: "#dcb980", width: 1.3, profile: "flat", roughness: 0.68},
        walnut: {kind: "wood", texture: "Wood066", base: "#5f3f28", width: 1.6, profile: "scoop", roughness: 0.5},
        mahogany: {kind: "wood", texture: "Wood026", base: "#6e2c1c", width: 1.8, profile: "scoop", roughness: 0.38},
        ebony: {kind: "wood", texture: "Wood051", base: "#26211e", width: 1.2, profile: "flat", roughness: 0.35},
        whitewashed: {kind: "wood", texture: "Wood049", base: "#dfd9cc", width: 1.5, profile: "flat", roughness: 0.75},
        "old-gold": {kind: "gold", width: 2.2, profile: "stepped", roughness: 0.42},
        "baroque-gold": {kind: "gold", width: 3.4, profile: "baroque", roughness: 0.36, ornate: true},
    };
    const PROFILES = {
        flat: [[0, 0], [0, 0.22], [0.06, 0.3], [0.9, 0.34], [1, 0.27], [1, -0.45]],
        scoop: curve([[0, 0], [0, 0.2]], (t) => [0.04 + 0.7 * t, 0.2 + 0.26 * t * t], [[0.86, 0.5], [1, 0.42], [1, -0.45]]),
        stepped: curve([[0, 0], [0, 0.12], [0.07, 0.14]], (t) => [0.07 + 0.46 * t, 0.14 + 0.2 * t * t],
            [[0.58, 0.36], [0.62, 0.3], ...arc(0.76, 0.3, 0.14, 0.13), [1, 0.22], [1, -0.4]]),
        baroque: [[0, 0], [0, 0.07], ...arc(0.055, 0.07, 0.055, 0.05), [0.14, 0.09],
            ...arc(0.42, 0.1, 0.28, 0.2), ...arc(0.76, 0.14, 0.06, 0.05), ...arc(0.91, 0.12, 0.09, 0.09), [1, -0.3]],
    };
    const MATTE_WALL = "#e7e1d6";
    // The mean color of each wood texture, to tint it to the color of a frame
    const WOOD_MEANS = {Wood051: "#41342e", Wood049: "#7b634e", Wood066: "#593413", Wood026: "#683917"};

    function arc(center, height, radiusX, radiusZ) {
        const points = [];
        for (let i = 0; i <= 8; i++) {
            const angle = Math.PI * (1 - i / 8);
            points.push([center + radiusX * Math.cos(angle), height + radiusZ * Math.sin(angle)]);
        }
        return points;
    }

    function curve(before, shape, after) {
        const points = [...before];
        for (let i = 0; i <= 8; i++) points.push(shape(i / 8));
        return [...points, ...after];
    }

    function random(seed) {
        return () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    }

    function drawing(width, height, draw) {
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        draw(canvas.getContext("2d"), width, height);
        return canvas;
    }

    // A repeating scroll with leaves, drawn in the band of the moulding which faces the viewer
    function scrolls(context, width, height, style) {
        const motifs = 4, top = height * 0.2, bottom = height * 0.68, middle = (top + bottom) / 2;
        const size = (bottom - top) / 2;
        context.strokeStyle = style;
        context.fillStyle = style;
        context.lineCap = "round";
        for (let motif = 0; motif < motifs; motif++) {
            const x = (motif + 0.5) * width / motifs, step = width / motifs;
            for (const side of [-1, 1]) {
                context.lineWidth = size * 0.17;
                context.beginPath();
                for (let t = 0; t <= 1; t += 0.03) {
                    const angle = t * 3.4 * Math.PI, radius = size * 0.75 * (1 - 0.72 * t);
                    const px = x + side * (step * 0.26 + radius * Math.cos(angle)), py = middle + radius * Math.sin(angle);
                    t ? context.lineTo(px, py) : context.moveTo(px, py);
                }
                context.stroke();
                for (let leaf = 0; leaf < 3; leaf++) {
                    context.beginPath();
                    context.ellipse(x + side * (step * 0.06 + leaf * size * 0.14), middle + (leaf - 1) * size * 0.46,
                        size * 0.26, size * 0.1, side * (leaf - 1) * 0.8, 0, 2 * Math.PI);
                    context.fill();
                }
            }
            context.beginPath();
            context.arc(x, middle, size * 0.2, 0, 2 * Math.PI);
            context.fill();
        }
        // Rows of beads along both sides of the scrolls
        for (const y of [top - size * 0.14, bottom + size * 0.14]) {
            for (let bead = 0; bead < 56; bead++) {
                context.beginPath();
                context.arc((bead + 0.5) * width / 56, y, size * 0.075, 0, 2 * Math.PI);
                context.fill();
            }
        }
    }

    function goldColor(context, width, height, spec, rand) {
        context.fillStyle = "#d7ad58";
        context.fillRect(0, 0, width, height);
        // Tarnish, then the red clay under the gilding where it has worn away
        for (let spot = 0; spot < 420; spot++) {
            const worn = rand() < 0.14;
            context.globalAlpha = worn ? 0.32 : 0.05 + rand() * 0.11;
            context.fillStyle = worn ? "#8b3b24" : rand() < 0.5 ? "#6d4f20" : "#f0d48a";
            context.beginPath();
            const radius = worn ? 1 + rand() * 4 : 4 + rand() * 26;
            context.ellipse(rand() * width, rand() * height, radius * (1 + rand() * 2), radius, 0, 0, 2 * Math.PI);
            context.fill();
        }
        context.globalAlpha = 1;
        // Grime in the recess along the picture
        const grime = context.createLinearGradient(0, 0, 0, height * 0.2);
        grime.addColorStop(0, "rgba(38, 24, 8, 0.75)");
        grime.addColorStop(1, "rgba(38, 24, 8, 0)");
        context.fillStyle = grime;
        context.fillRect(0, 0, width, height * 0.2);
        if (spec.ornate) {
            // Darker around the relief, as in the hollows of an old carving
            context.filter = "blur(5px)";
            scrolls(context, width, height, "rgba(48, 28, 6, 0.5)");
            context.filter = "none";
        }
    }

    function goldTextures(THREE, name, spec, maxAnisotropy) {
        const rand = random(Object.keys(FRAMES).indexOf(name) * 7919 + 17);
        const color = drawing(1024, 512, (context, width, height) => {
            goldColor(context, width, height, spec, rand);
        });
        const bump = drawing(1024, 512, (context, width, height) => {
            context.fillStyle = "#606060";
            context.fillRect(0, 0, width, height);
            if (spec.ornate) {
                context.filter = "blur(2px)";
                scrolls(context, width, height, "#ffffff");
                context.filter = "none";
            }
        });
        const textures = {color: new THREE.CanvasTexture(color), bump: new THREE.CanvasTexture(bump)};
        textures.color.colorSpace = THREE.SRGBColorSpace;
        for (const texture of Object.values(textures)) {
            texture.wrapS = THREE.RepeatWrapping;
            texture.anisotropy = maxAnisotropy;
        }
        return textures;
    }

    // The moulding: its profile swept along the 4 sides, which meet in mitres at the corners
    function frameGeometry(THREE, width, height, frameWidth, profile, repeatLength) {
        const scaled = profile.map(([out, up]) => [out * frameWidth, up * frameWidth]);
        const lengths = [0];
        for (let i = 1; i < scaled.length; i++) {
            lengths.push(lengths[i - 1] + Math.hypot(scaled[i][0] - scaled[i - 1][0], scaled[i][1] - scaled[i - 1][1]));
        }
        // Each side goes counterclockwise seen from the front. Its own vertices keep the mitres sharp
        const sides = [[-1, -1, 1, -1], [1, -1, 1, 1], [1, 1, -1, 1], [-1, 1, -1, -1]];
        const positions = [], uvs = [], indices = [];
        sides.forEach(([x0, y0, x1, y1]) => {
            const first = positions.length / 3;
            const along = x0 !== x1 ? 0 : 1;
            for (const [corner, cx, cy] of [[0, x0, y0], [1, x1, y1]]) {
                scaled.forEach(([out, up], ind) => {
                    const x = cx * (width / 2 + out), y = cy * (height / 2 + out);
                    positions.push(x, y, up);
                    // V goes from the picture to the outside, whatever the side
                    uvs.push((along === 0 ? x : y) / repeatLength, 1 - lengths[ind] / lengths[lengths.length - 1]);
                });
            }
            const count = scaled.length;
            for (let i = 0; i < count - 1; i++) {
                const a = first + i, b = a + 1, c = first + count + i, d = c + 1;
                indices.push(a, b, c, b, d, c);
            }
        });
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
        geometry.setIndex(indices);
        geometry.computeVertexNormals();
        return geometry;
    }

    class Viewer {
        constructor(THREE, OrbitControls, RoomEnvironment, container) {
            this.THREE = THREE;
            this.container = container;
            this.renderer = new THREE.WebGLRenderer({antialias: true});
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.renderer.shadowMap.enabled = true;
            this.renderer.shadowMap.type = THREE.PCFShadowMap;
            container.appendChild(this.renderer.domElement);

            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(MATTE_WALL);
            const environment = new THREE.PMREMGenerator(this.renderer);
            this.scene.environment = environment.fromScene(new RoomEnvironment(), 0.04).texture;
            this.scene.environmentIntensity = 0.5;
            environment.dispose();

            this.light = new THREE.DirectionalLight("#fff4e4", 1.0);
            this.light.castShadow = true;
            this.light.shadow.mapSize.set(2048, 2048);
            this.light.shadow.radius = 6;
            this.scene.add(this.light, this.light.target);

            // Only the shadows are drawn: the wall itself is the background, so that it has no edge nor lighting
            this.wall = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShadowMaterial({opacity: 0.32}));
            this.wall.receiveShadow = true;
            this.scene.add(this.wall);

            this.object = new THREE.Group();
            this.scene.add(this.object);
            this.camera = new THREE.PerspectiveCamera(28, 1, 0.1, 2000);
            this.controls = new OrbitControls(this.camera, this.renderer.domElement);
            Object.assign(this.controls, {
                enablePan: false, minAzimuthAngle: -1.2, maxAzimuthAngle: 1.2,
                minPolarAngle: 0.85, maxPolarAngle: 2.3,
            });
            this.controls.addEventListener("change", () => this.render());
            this.textures = {};
            new ResizeObserver(() => this.resize()).observe(container);
        }

        resize() {
            const {clientWidth: width, clientHeight: height} = this.container;
            if (!width || !height) return;
            this.renderer.setSize(width, height);
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            if (this.fit) this.frameCamera();
            this.render();
        }

        render() {
            this.renderer.render(this.scene, this.camera);
        }

        // Distance showing the whole framed object, in any shape of the view
        frameCamera() {
            const {width, height} = this.fit;
            const halfFov = this.camera.fov * Math.PI / 360;
            const distance = 1.22 * Math.max(height / 2, width / 2 / this.camera.aspect) / Math.tan(halfFov);
            this.camera.position.set(distance * 0.36, distance * 0.05, distance * 0.93);
            this.controls.target.set(0, 0, 0);
            this.controls.minDistance = distance * 0.25;
            this.controls.maxDistance = distance * 1.8;
            this.controls.update();
        }

        // Load the photographed wood of a frame, the only part of the viewer coming from files
        async prepare(frameName) {
            const spec = FRAMES[frameName];
            if (spec?.kind !== "wood" || this.textures[spec.texture]) return;
            const THREE = this.THREE, loader = new THREE.TextureLoader();
            const [color, normal] = await Promise.all(["color", "normal"].map((kind) =>
                loader.loadAsync(`${WOOD}${spec.texture}_${kind}.jpg`)));
            color.colorSpace = THREE.SRGBColorSpace;
            for (const texture of [color, normal]) {
                texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
                texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
            }
            this.textures[spec.texture] = {color, normal};
        }

        material(name, spec) {
            const THREE = this.THREE;
            if (spec.kind === "wood") {
                const {color: map, normal: normalMap} = this.textures[spec.texture];
                // The ratio of the frame color to the mean of the texture, in linear colors. It can be above 1
                const base = new THREE.Color(spec.base), mean = new THREE.Color(WOOD_MEANS[spec.texture]);
                const color = new THREE.Color(base.r / mean.r, base.g / mean.g, base.b / mean.b);
                return new THREE.MeshStandardMaterial({
                    map, normalMap, color, roughness: spec.roughness, normalScale: new THREE.Vector2(0.6, 0.6),
                });
            }
            this.textures[name] ??= goldTextures(THREE, name, spec, this.renderer.capabilities.getMaxAnisotropy());
            const {color, bump} = this.textures[name];
            return new THREE.MeshStandardMaterial({
                map: color, bumpMap: bump, bumpScale: spec.ornate ? 5 : 0.5, roughness: spec.roughness, metalness: 0.92,
                // Gold is seen through what it reflects
                envMapIntensity: 2.6,
            });
        }

        clear() {
            this.object.traverse((child) => {
                child.geometry?.dispose();
                for (const material of [child.material].flat()) {
                    if (material?.map?.userData.picture) material.map.dispose();
                    material?.dispose();
                }
            });
            this.object.clear();
        }

        show(image, frameName, size) {
            const THREE = this.THREE, spec = FRAMES[frameName];
            this.clear();
            const [pageWidth, pageHeight] = size.page, [pictureWidth, pictureHeight] = size.picture;
            const texture = new THREE.Texture(image);
            texture.userData.picture = true;
            texture.colorSpace = THREE.SRGBColorSpace;
            texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
            texture.needsUpdate = true;
            const paintedLayer = new THREE.Mesh(new THREE.PlaneGeometry(pictureWidth, pictureHeight),
                new THREE.MeshBasicMaterial({map: texture}));
            paintedLayer.position.z = 0.012;
            // Not lit, so that it shows the colors to paint: only the shadow of the frame falls on it
            const shadowLayer = new THREE.Mesh(new THREE.PlaneGeometry(pageWidth, pageHeight),
                new THREE.ShadowMaterial({opacity: 0.4}));
            shadowLayer.position.z = 0.024;
            shadowLayer.receiveShadow = true;
            this.object.add(paintedLayer, shadowLayer);

            let outerWidth = pageWidth, outerHeight = pageHeight, depth = 0.75;
            const white = new THREE.MeshBasicMaterial({color: "#f6f3ec"});
            if (spec) {
                const frameWidth = spec.width * Math.min(1.5, Math.max(0.55, Math.max(pageWidth, pageHeight) / 20));
                const profile = PROFILES[spec.profile];
                depth = -Math.min(...profile.map(([, up]) => up)) * frameWidth;
                const page = new THREE.Mesh(new THREE.PlaneGeometry(pageWidth, pageHeight), white);
                const moulding = new THREE.Mesh(frameGeometry(THREE, pageWidth, pageHeight, frameWidth, profile,
                    spec.ornate ? 2.8 * frameWidth : 16), this.material(frameName, spec));
                moulding.castShadow = moulding.receiveShadow = true;
                this.object.add(page, moulding);
                outerWidth += 2 * frameWidth;
                outerHeight += 2 * frameWidth;
            } else {
                // A canvas stretched on its bars, with no frame
                const stretched = new THREE.Mesh(new THREE.BoxGeometry(pageWidth, pageHeight, depth), white);
                stretched.position.z = -depth / 2;
                stretched.castShadow = true;
                // Lit on its sides, and index 4 is the front
                stretched.material = Array(6).fill(new THREE.MeshStandardMaterial({color: "#e8e3d8", roughness: 0.9}));
                stretched.material[4] = white;
                this.object.add(stretched);
            }

            const largest = Math.max(outerWidth, outerHeight);
            this.wall.position.z = -depth;
            this.wall.scale.set(largest * 8, largest * 8, 1);
            this.light.position.set(-largest * 0.7, largest * 0.9, largest * 1.3);
            const shadow = this.light.shadow.camera;
            shadow.left = shadow.bottom = -largest;
            shadow.right = shadow.top = largest;
            shadow.far = largest * 5;
            shadow.updateProjectionMatrix();

            const fit = {width: outerWidth, height: outerHeight};
            // Keep the point of view while only the frame changes
            if (!this.fit || this.fit.width !== fit.width || this.fit.height !== fit.height) {
                this.fit = fit;
                this.resize();
                this.frameCamera();
            }
            this.render();
            this.container.dataset.rendered = frameName;
        }
    }

    let viewerPromise = null;

    function viewer(container) {
        viewerPromise ??= Promise.all([import("three"), import(VENDOR + "OrbitControls.js"),
            import(VENDOR + "RoomEnvironment.js")]).then(([THREE, orbit, room]) =>
            new Viewer(THREE, orbit.OrbitControls, room.RoomEnvironment, container));
        return viewerPromise;
    }

    let latest = 0;

    function update(detail, frame, size) {
        const hidden = {display: "none"};
        const picture = detail && document.getElementById(`preview-image-${detail}`);
        if (!picture || !size) return hidden;
        const request = ++latest;
        const container = document.getElementById("frame-view");
        const message = document.getElementById("frame-message");
        // After the update of the page, so that the viewer gets the size of its now visible container
        requestAnimationFrame(() => viewer(container).then(async (shown) => {
            const image = new Image();
            image.src = picture.src;
            await image.decode();
            await shown.prepare(frame);
            if (request === latest) shown.show(image, frame, size);
        }).catch((problem) => {
            console.warn("The 3D preview can't be shown", problem);
            message.textContent = "The 3D preview can't be shown in this browser, the files are not affected.";
        }));
        return {};
    }

    window.dash_clientside = Object.assign({}, window.dash_clientside, {frame3d: {update}});
})();
