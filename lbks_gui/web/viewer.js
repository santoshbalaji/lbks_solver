import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';
import URDFLoader from './vendor/urdf-loader/src/URDFLoader.js';

// All positions use the URDF's Z-up coordinate system, in metres.
export class RobotView {
  constructor(container, onTarget, onRelease, onStatus) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#101b2a');
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    container.append(this.renderer.domElement);
    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.enableDamping = true;
    this.resetCamera();

    this.scene.add(new THREE.HemisphereLight(0xeaf4ff, 0x344a61, 2.5));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(2, -3, 5);
    light.castShadow = true;
    light.shadow.mapSize.set(2048, 2048);
    this.scene.add(light);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8),
      new THREE.MeshStandardMaterial({ color: 0x142232, roughness: 1 }));
    floor.position.z = -0.012;
    floor.receiveShadow = true;
    this.scene.add(floor);
    const grid = new THREE.GridHelper(4, 40, 0x33485b, 0x24384b);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.008;
    this.scene.add(grid, new THREE.AxesHelper(0.22));

    const sphere = color => new THREE.Mesh(new THREE.SphereGeometry(0.018, 20, 12),
      new THREE.MeshBasicMaterial({ color }));
    this.target = sphere(0x68bef7);
    this.actual = sphere(0x68e8ad);
    this.errorLine = new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xe7b667 }));
    this.scene.add(this.target, this.actual, this.errorLine);
    this.transform = new TransformControls(this.camera, this.renderer.domElement);
    this.transform.setMode('translate');
    this.transform.setSize(0.65);
    this.transform.attach(this.target);
    this.scene.add(this.transform.getHelper());
    this.transform.addEventListener('dragging-changed', event => {
      this.orbit.enabled = !event.value;
    });
    this.transform.addEventListener('objectChange', () => {
      this.updateLine();
      onTarget(this.target.position.toArray());
    });
    this.transform.addEventListener('mouseUp', () => onRelease(this.target.position.toArray()));
    this.obstacles = new THREE.Group();
    this.scene.add(this.obstacles);
    this.lastJoints = {};
    this.collisionVisible = false;

    const manager = new THREE.LoadingManager();
    let failed = false;
    manager.onError = url => { failed = true; onStatus(`Mesh could not load: ${url}`); };
    manager.onLoad = () => {
      this.robot.traverse(object => {
        if (object.isMesh) { object.castShadow = true; object.receiveShadow = true; }
        if (object.isURDFCollider) {
          object.visible = this.collisionVisible;
          object.traverse(child => {
            if (child.isMesh) child.material = new THREE.MeshBasicMaterial({
              color: 0xffbd73, wireframe: true, transparent: true, opacity: 0.5,
            });
          });
        }
      });
      if (!failed) onStatus('URDF loaded · drag the blue target to explore');
    };
    const loader = new URDFLoader(manager);
    loader.packages = { lbks_description: '/robot-assets' };
    loader.parseCollision = true;
    loader.load('/robot.urdf', robot => {
      this.robot = robot;
      this.scene.add(robot);
      robot.setJointValues(this.lastJoints);
    }, undefined, error => onStatus(`URDF error: ${error.message}`));

    new ResizeObserver(() => {
      const width = container.clientWidth, height = container.clientHeight;
      this.camera.aspect = width / Math.max(height, 1);
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
    }).observe(container);
    this.renderer.setAnimationLoop(() => {
      this.orbit.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  resetCamera() {
    this.camera.position.set(1.65, -1.9, 1.25);
    this.orbit.target.set(0.25, 0, 0.38);
    this.orbit.update();
  }

  setPose(names, joints, actual) {
    this.lastJoints = Object.fromEntries(names.map((name, index) => [name, joints[index]]));
    this.robot?.setJointValues(this.lastJoints);
    this.actual.position.fromArray(actual);
    this.updateLine();
  }

  setTarget(target) { this.target.position.fromArray(target); this.updateLine(); }
  updateLine() { this.errorLine.geometry.setFromPoints([this.target.position, this.actual.position]); }

  showCollision(enabled) {
    this.collisionVisible = enabled;
    this.robot?.traverse(object => { if (object.isURDFCollider) object.visible = enabled; });
  }

  // Extension point for a future collision checker: boxes in the URDF root frame.
  setObstacles(boxes) {
    this.obstacles.children.forEach(mesh => { mesh.geometry.dispose(); mesh.material.dispose(); });
    this.obstacles.clear();
    for (const box of boxes) {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(...box.size),
        new THREE.MeshStandardMaterial({ color: box.color ?? 0xffa56d, transparent: true, opacity: 0.35 }));
      mesh.position.fromArray(box.position);
      this.obstacles.add(mesh);
    }
  }
}
