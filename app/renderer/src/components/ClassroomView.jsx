import React, { useEffect, useRef } from "react";
import Phaser from "phaser";

/*
  Unified school map with walking character sprites.
  Canvas: 800x750
    [0,   0] - [800, 620]: Reference image (classroom_bg.png scaled)
    [0, 620] - [800, 750]: Playground (drawn programmatically)

  The reference image contains classroom (top-left), office (top-right),
  hallway with lockers (bottom). The playground seamlessly extends below.
*/

// ── Exact color palette from the reference image ─────────────────────────
const PAL = {
  border: 0x3b3558,
  gold: 0x8b7a4a,
  white: 0xffffff,
  cream: 0xf0e0c0,
  floorLight: 0xc8b89a,
  floorDark: 0xbaa882,
  deskWood: 0xc49560,
  deskEdge: 0x9a7040,
  chairGreen: 0x2d6e5a,
  chairDark: 0x1e4d3d,
  lockerBlue: 0x8899bb,
  lockerDark: 0x7788aa,
  bookshelf: 0x8b6530,
  windowGlass: 0x88bbdd,
  windowFrame: 0xddccaa,
  curtainRed: 0xaa3333,
  doorBrown: 0x9a7040,
  grassA: 0x6a9a3a,
  grassB: 0x5a8a2a,
  treeLeafA: 0x4a8a3a,
  treeLeafB: 0x3a7a2a,
  trunk: 0x8b6530,
  sandbox: 0xd8c8a0,
  hoopPole: 0x8899bb,
  hoopRim: 0xd08040,
  skinTone: 0xf0c8a0,
  pants: 0x334455,
  eyes: 0x222222,
};

// ── Area bounding boxes on 800x750 ───────────────────────────────────────
const AREAS = {
  classroom:  { x: 0,   y: 0,   w: 530, h: 440, label: "교실",   color: 0x60a5fa },
  office:     { x: 530, y: 0,   w: 270, h: 440, label: "교무실", color: 0xa855f7 },
  hallway:    { x: 0,   y: 440, w: 800, h: 180, label: "복도",   color: 0x4ade80 },
  playground: { x: 0,   y: 620, w: 800, h: 130, label: "운동장", color: 0xf59e0b },
};

// ── Student/teacher positions per area ───────────────────────────────────
const AREA_POSITIONS = {
  classroom: {
    // 30 desks — 5 columns × 6 rows, denser grid to fit 30 students.
    // Smaller spacing (~62px) + reduced sprite size keeps them within the
    // classroom floor of the reference background (x≈120-420, y≈160-500).
    desks: [
      { x: 138, y: 168 }, { x: 200, y: 168 }, { x: 262, y: 168 }, { x: 324, y: 168 }, { x: 386, y: 168 },
      { x: 138, y: 230 }, { x: 200, y: 230 }, { x: 262, y: 230 }, { x: 324, y: 230 }, { x: 386, y: 230 },
      { x: 138, y: 292 }, { x: 200, y: 292 }, { x: 262, y: 292 }, { x: 324, y: 292 }, { x: 386, y: 292 },
      { x: 138, y: 354 }, { x: 200, y: 354 }, { x: 262, y: 354 }, { x: 324, y: 354 }, { x: 386, y: 354 },
      { x: 138, y: 416 }, { x: 200, y: 416 }, { x: 262, y: 416 }, { x: 324, y: 416 }, { x: 386, y: 416 },
      { x: 138, y: 478 }, { x: 200, y: 478 }, { x: 262, y: 478 }, { x: 324, y: 478 }, { x: 386, y: 478 },
    ],
    teacher: { x: 280, y: 128 },
  },
  office: {
    seats: [{ x: 600, y: 280 }, { x: 680, y: 320 }],
    teacher: { x: 700, y: 240 },
  },
  hallway: {
    // 32 slots (≥30) — 4 rows × 8 columns, no wrap for 30 students.
    path: [
      { x: 80, y: 500 }, { x: 175, y: 500 }, { x: 270, y: 500 }, { x: 365, y: 500 },
      { x: 460, y: 500 }, { x: 555, y: 500 }, { x: 650, y: 500 }, { x: 740, y: 500 },
      { x: 80, y: 538 }, { x: 175, y: 538 }, { x: 270, y: 538 }, { x: 365, y: 538 },
      { x: 460, y: 538 }, { x: 555, y: 538 }, { x: 650, y: 538 }, { x: 740, y: 538 },
      { x: 80, y: 576 }, { x: 175, y: 576 }, { x: 270, y: 576 }, { x: 365, y: 576 },
      { x: 460, y: 576 }, { x: 555, y: 576 }, { x: 650, y: 576 }, { x: 740, y: 576 },
      { x: 80, y: 610 }, { x: 175, y: 610 }, { x: 270, y: 610 }, { x: 365, y: 610 },
      { x: 460, y: 610 }, { x: 555, y: 610 }, { x: 650, y: 610 }, { x: 740, y: 610 },
    ],
    teacher: { x: 400, y: 555 },
  },
  playground: {
    // 30 scattered positions across the playground band (y≈645-735).
    scattered: [
      { x: 70, y: 655 }, { x: 160, y: 670 }, { x: 250, y: 650 }, { x: 340, y: 668 },
      { x: 430, y: 652 }, { x: 520, y: 670 }, { x: 610, y: 650 }, { x: 700, y: 668 },
      { x: 110, y: 690 }, { x: 200, y: 700 }, { x: 290, y: 688 }, { x: 380, y: 702 },
      { x: 470, y: 690 }, { x: 560, y: 700 }, { x: 650, y: 688 }, { x: 740, y: 700 },
      { x: 70, y: 722 }, { x: 160, y: 730 }, { x: 250, y: 720 }, { x: 340, y: 732 },
      { x: 430, y: 722 }, { x: 520, y: 730 }, { x: 610, y: 720 }, { x: 700, y: 730 },
      { x: 120, y: 660 }, { x: 300, y: 712 }, { x: 490, y: 660 }, { x: 580, y: 712 },
      { x: 380, y: 660 }, { x: 660, y: 705 },
    ],
    teacher: { x: 400, y: 685 },
  },
};

// ── Character profiles ───────────────────────────────────────────────────
const CHARACTER_PROFILES = [
  { id: "teacher", hair: 0x553320, shirt: 0xeeeeee, isTeacher: true },
  { id: "S01", hair: 0x553320, shirt: 0x4477aa },
  { id: "S02", hair: 0x886640, shirt: 0x44aa77 },
  { id: "S03", hair: 0x332244, shirt: 0xaa6644 },
  { id: "S04", hair: 0x221122, shirt: 0x5588bb },
  { id: "S05", hair: 0x664422, shirt: 0x66aa55 },
  { id: "S06", hair: 0x443322, shirt: 0xbb5555 },
  { id: "S07", hair: 0x554433, shirt: 0x7766aa },
  { id: "S08", hair: 0x332211, shirt: 0x44bbaa },
  { id: "S09", hair: 0x665544, shirt: 0xaa7744 },
  { id: "S10", hair: 0x222233, shirt: 0x5599aa },
  { id: "S11", hair: 0x776655, shirt: 0x448866 },
  { id: "S12", hair: 0x443355, shirt: 0xbb7766 },
  { id: "S13", hair: 0x554422, shirt: 0x6688bb },
  { id: "S14", hair: 0x332233, shirt: 0x77aa55 },
  { id: "S15", hair: 0x665533, shirt: 0xaa5577 },
  { id: "S16", hair: 0x443311, shirt: 0x55aaaa },
  { id: "S17", hair: 0x221133, shirt: 0x88bb55 },
  { id: "S18", hair: 0x554411, shirt: 0xbb8844 },
  { id: "S19", hair: 0x333344, shirt: 0x5577bb },
  { id: "S20", hair: 0x664433, shirt: 0x44aa88 },
  { id: "S21", hair: 0x443322, shirt: 0x6699aa },
  { id: "S22", hair: 0x665544, shirt: 0xaa6655 },
  { id: "S23", hair: 0x332244, shirt: 0x559977 },
  { id: "S24", hair: 0x554433, shirt: 0xbb7755 },
  { id: "S25", hair: 0x221133, shirt: 0x5588aa },
  { id: "S26", hair: 0x664422, shirt: 0x77aa66 },
  { id: "S27", hair: 0x443355, shirt: 0xaa5566 },
  { id: "S28", hair: 0x554422, shirt: 0x6688aa },
  { id: "S29", hair: 0x332233, shirt: 0x88aa66 },
  { id: "S30", hair: 0x665533, shirt: 0xbb7744 },
];

// ── Derive active area from scenario + action ────────────────────────────
function deriveArea(scenario, action, turn, totalTurns) {
  if (action === "private_correction") return "office";
  if (!scenario) return "classroom";
  if (scenario === "recess_to_math") {
    const pivot = totalTurns ? totalTurns / 2 : 15;
    if (turn <= pivot) return "playground";
    if (turn <= pivot + 2) return "hallway";
    return "classroom";
  }
  return "classroom";
}

// ── Student risk color ───────────────────────────────────────────────────
function studentColor(student) {
  if (student.is_managed) return 0xa855f7;
  if (student.is_identified) return 0x60a5fa;
  const s = student.state || {};
  const esc = s.escalation ?? s.escalation_risk ?? 0;
  const dis = s.distress ?? s.distress_level ?? 0;
  if (esc > 0.6 || dis > 0.7) return 0xef4444;
  if (esc > 0.35 || dis > 0.45) return 0xf59e0b;
  return 0x4ade80;
}

// ── Draw a detailed pixel-art character ──────────────────────────────────
function drawCharacter(g, x, y, hairColor, shirtColor, facing, frame, isTeacher) {
  const S = isTeacher ? 2.2 : 1.7;  // base pixel size (shrunk for 30-student density)

  // Shadow
  g.fillStyle(0x000000, 0.25);
  g.fillEllipse(x, y + S * 14, S * 8, S * 3);

  // ── Legs + Shoes ──
  const shoeColor = 0x443322;
  g.fillStyle(PAL.pants);
  if (frame === 0) {
    // standing / walk frame A
    g.fillRect(x - S * 3, y + S * 9, S * 2.5, S * 4);
    g.fillRect(x + S * 0.5, y + S * 9, S * 2.5, S * 4);
    g.fillStyle(shoeColor);
    g.fillRect(x - S * 3, y + S * 12.5, S * 3, S * 1.5);
    g.fillRect(x + S * 0.5, y + S * 12.5, S * 3, S * 1.5);
  } else {
    // walk frame B
    g.fillRect(x - S * 2, y + S * 9, S * 2.5, S * 5);
    g.fillRect(x + S * 1, y + S * 9, S * 2.5, S * 3.5);
    g.fillStyle(shoeColor);
    g.fillRect(x - S * 2, y + S * 13.5, S * 3, S * 1.5);
    g.fillRect(x + S * 1, y + S * 12, S * 3, S * 1.5);
  }

  // ── Body / Shirt ──
  g.fillStyle(shirtColor);
  g.fillRect(x - S * 4, y + S * 2, S * 8, S * 7.5);
  // Shirt collar / neckline
  const collarColor = isTeacher ? 0xffffff : ((shirtColor & 0xfefefe) >> 1) + 0x808080;
  g.fillStyle(collarColor, 0.6);
  g.fillRect(x - S * 1.5, y + S * 1.5, S * 3, S * 1);

  // ── Arms ──
  g.fillStyle(shirtColor);
  // Left arm
  g.fillRect(x - S * 5.5, y + S * 3, S * 2, S * 5);
  // Right arm
  g.fillRect(x + S * 3.5, y + S * 3, S * 2, S * 5);
  // Hands (skin)
  g.fillStyle(PAL.skinTone);
  g.fillRect(x - S * 5.5, y + S * 7.5, S * 2, S * 1.5);
  g.fillRect(x + S * 3.5, y + S * 7.5, S * 2, S * 1.5);

  // ── Head ──
  g.fillStyle(PAL.skinTone);
  g.fillRect(x - S * 4, y - S * 7, S * 8, S * 9);

  // ── Hair ──
  g.fillStyle(hairColor);
  // Top of head
  g.fillRect(x - S * 4.5, y - S * 8, S * 9, S * 4);
  if (facing === "up") {
    g.fillRect(x - S * 4.5, y - S * 8, S * 9, S * 9);
  } else if (facing === "left") {
    g.fillRect(x - S * 4.5, y - S * 8, S * 4, S * 8);
    g.fillRect(x - S * 4.5, y - S * 8, S * 9, S * 4);
  } else if (facing === "right") {
    g.fillRect(x + S * 0.5, y - S * 8, S * 4, S * 8);
    g.fillRect(x - S * 4.5, y - S * 8, S * 9, S * 4);
  } else {
    // down — show bangs
    g.fillRect(x - S * 4.5, y - S * 8, S * 9, S * 4);
    g.fillRect(x - S * 4.5, y - S * 8, S * 2, S * 6);
    g.fillRect(x + S * 2.5, y - S * 8, S * 2, S * 6);
  }

  // ── Face (visible when not facing up) ──
  if (facing !== "up") {
    const fOff = facing === "left" ? -S * 1 : facing === "right" ? S * 1 : 0;
    // Eyes
    g.fillStyle(0xffffff);
    g.fillRect(x - S * 2.5 + fOff, y - S * 3, S * 2, S * 2);
    g.fillRect(x + S * 0.5 + fOff, y - S * 3, S * 2, S * 2);
    g.fillStyle(PAL.eyes);
    g.fillRect(x - S * 2 + fOff, y - S * 2.5, S * 1.2, S * 1.2);
    g.fillRect(x + S * 1 + fOff, y - S * 2.5, S * 1.2, S * 1.2);
    // Mouth
    g.fillStyle(0xcc8866);
    g.fillRect(x - S * 0.5 + fOff, y + S * 0.5, S * 1, S * 0.5);
    // Cheek blush
    g.fillStyle(0xffaaaa, 0.3);
    g.fillRect(x - S * 3 + fOff, y - S * 0.5, S * 1.5, S * 1);
    g.fillRect(x + S * 1.5 + fOff, y - S * 0.5, S * 1.5, S * 1);
  }

  // ── Teacher: glasses + tie ──
  if (isTeacher) {
    if (facing !== "up") {
      const fOff = facing === "left" ? -S * 1 : facing === "right" ? S * 1 : 0;
      g.lineStyle(1, 0x444444, 0.8);
      g.strokeRect(x - S * 3 + fOff, y - S * 3.5, S * 2.5, S * 2.5);
      g.strokeRect(x + S * 0.5 + fOff, y - S * 3.5, S * 2.5, S * 2.5);
      g.lineStyle(0);
    }
    // Tie
    g.fillStyle(0xcc3333);
    g.fillTriangle(x - S * 0.8, y + S * 2, x + S * 0.8, y + S * 2, x, y + S * 5);
  }
}

// ── Facing direction from movement vector ────────────────────────────────
function getFacing(dx, dy) {
  if (Math.abs(dx) > Math.abs(dy)) {
    return dx > 0 ? "right" : "left";
  }
  return dy > 0 ? "down" : "up";
}

// ── Phaser Scene ─────────────────────────────────────────────────────────
class UnifiedSchoolScene extends Phaser.Scene {
  constructor() {
    super("UnifiedSchoolScene");
    this.mode = "classic";
    this.students = [];
    this.targetStudentId = null;
    this.activeArea = "classroom";

    // Graphics layers
    this.bgImage = null;
    this.playgroundGraphics = null;
    this.overlayGraphics = null;
    this.highlightGraphics = null;
    this.characterGraphics = null;

    // Character state: map id -> { x, y, targetX, targetY, hair, shirt, facing, frame, frameCounter, isTeacher }
    this.characters = new Map();
    this._frameCount = 0;

    // Interaction lines (v2): [{x1,y1,x2,y2,color,text,alpha,createdAt}]
    this._interactionLines = [];
    this.interactionGraphics = null;

    // UI
    this.statusText = null;
    this.turnText = null;
    this.actionText = null;
    this.actionBg = null;
    this.bubbleGroup = null;
    this.areaLabelTexts = {};
  }

  preload() {
    this.load.image("classroom", "/assets/concept_motion/imagegen/school_background_imagegen.png");
    this.load.image("playground", "/assets/playground_bg.png");

    // motion sheets (5 characters × 6 frames each, 192x192 per frame)
    this.load.spritesheet("char_01", "/assets/concept_motion/imagegen/sheets/concept_01_boy_uniform_motion.png", { frameWidth: 192, frameHeight: 192 });
    this.load.spritesheet("char_02", "/assets/concept_motion/imagegen/sheets/concept_02_girl_sailor_motion.png", { frameWidth: 192, frameHeight: 192 });
    this.load.spritesheet("char_03", "/assets/concept_motion/imagegen/sheets/concept_03_boy_blazer_motion.png", { frameWidth: 192, frameHeight: 192 });
    this.load.spritesheet("char_04", "/assets/concept_motion/imagegen/sheets/concept_04_girl_blonde_motion.png", { frameWidth: 192, frameHeight: 192 });
    this.load.spritesheet("char_05", "/assets/concept_motion/imagegen/sheets/concept_05_teacher_motion.png", { frameWidth: 192, frameHeight: 192 });
  }

  create() {
    const W = this.scale.width;   // 800
    const H = this.scale.height;  // 750

    // ── Sprite animations (idle + walk per character) ──
    for (let i = 1; i <= 5; i++) {
      const key = `char_0${i}`;
      if (!this.textures.exists(key)) continue;
      if (!this.anims.exists(`${key}_idle`)) {
        this.anims.create({ key: `${key}_idle`, frames: [{ key, frame: 0 }], frameRate: 1, repeat: -1 });
      }
      if (!this.anims.exists(`${key}_walk`)) {
        this.anims.create({ key: `${key}_walk`, frames: this.anims.generateFrameNumbers(key, { frames: [1, 2] }), frameRate: 6, repeat: -1 });
      }
    }

    // ── Layer 0: Background image (full opacity — this IS the world) ──
    this.bgImage = this.add.image(0, 0, "classroom").setOrigin(0, 0);
    this.bgImage.setDisplaySize(800, 620);

    // ── Layer 1: Programmatic playground (y=620 to y=750) ──
    this.playgroundGraphics = this.add.graphics();
    this._drawPlayground();

    // ── Layer 2: Dim overlays per area ──
    this.overlayGraphics = this.add.graphics();

    // ── Layer 3: Active area highlight glow ──
    this.highlightGraphics = this.add.graphics();

    // ── Layer 4: Character sprites (redrawn each frame) ──
    this.characterGraphics = this.add.graphics();

    // ── Layer 4.5: Interaction lines (v2) ──
    this.interactionGraphics = this.add.graphics();

    // ── Layer 5: Bubble group ──
    this.bubbleGroup = this.add.group();

    // ── Layer 6: Area labels ──
    this._createAreaLabels();

    // ── HUD: Status bar ──
    this.statusBg = this.add.graphics();
    this.statusBg.fillStyle(0x1e293b, 0.88);
    this.statusBg.fillRoundedRect(8, H - 42, 220, 30, 5);

    this.statusText = this.add.text(16, H - 36, "Waiting for session...", {
      fontSize: "10px",
      fontFamily: "'Pretendard', monospace",
      color: "#94a3b8",
    });

    // ── HUD: Turn counter ──
    this.turnBg = this.add.graphics();
    this.turnBg.fillStyle(0x1e293b, 0.88);
    this.turnBg.fillRoundedRect(W - 90, 8, 82, 22, 4);

    this.turnText = this.add.text(W - 82, 13, "", {
      fontSize: "10px",
      fontFamily: "monospace",
      color: "#e2e8f0",
    });

    // ── HUD: Action label ──
    this.actionBg = this.add.graphics();
    this.actionText = this.add.text(W / 2, 10, "", {
      fontSize: "11px",
      fontFamily: "'Pretendard', monospace",
      color: "#fbbf24",
      fontStyle: "bold",
    }).setOrigin(0.5, 0);

    // ── Transition label ──
    this.transitionLabel = this.add.text(W / 2, H / 2, "", {
      fontSize: "18px",
      fontFamily: "'Pretendard', monospace",
      color: "#ffffff",
      fontStyle: "bold",
      stroke: "#000000",
      strokeThickness: 3,
    }).setOrigin(0.5, 0.5).setAlpha(0);

    // Initialize characters at classroom positions
    this._initCharacters();

    // Draw initial overlays
    this._drawOverlaysAndHighlight("classroom");
  }

  // ── Pick a sprite texture key for a given character profile ──
  _charKeyFor(profileId, isTeacher) {
    if (isTeacher) return "char_05";
    // Map S01..S20 → char_01..char_04 (4 student sprites, round-robin)
    const m = /^S(\d+)$/.exec(profileId);
    const idx = m ? parseInt(m[1], 10) : 1;
    const slot = ((idx - 1) % 4) + 1; // 1..4
    return `char_0${slot}`;
  }

  // ── Create a Phaser sprite for a character, or null if textures missing ──
  _createCharSprite(profileId, isTeacher, x, y) {
    const key = this._charKeyFor(profileId, isTeacher);
    if (!this.textures.exists(key)) return null;
    try {
      const sprite = this.add.sprite(x, y, key, 0);
      // Frame is 192x192; shrunk for 30-student classroom density.
      const display = isTeacher ? 44 : 36;
      sprite.setDisplaySize(display, display);
      sprite.setOrigin(0.5, 0.85);
      sprite.setDepth(50);
      if (this.anims.exists(`${key}_idle`)) sprite.play(`${key}_idle`);
      return sprite;
    } catch (e) {
      return null;
    }
  }

  // ── Initialize character position data + sprites ──
  _initCharacters() {
    const desks = AREA_POSITIONS.classroom.desks;
    const teacherPos = AREA_POSITIONS.classroom.teacher;

    // Teacher — position data + sprite
    const teacherCh = {
      x: teacherPos.x, y: teacherPos.y,
      targetX: teacherPos.x, targetY: teacherPos.y,
      facing: "down", frame: 0, frameCounter: 0,
      isTeacher: true,
    };
    teacherCh.sprite = this._createCharSprite("teacher", true, teacherPos.x, teacherPos.y);
    teacherCh.charKey = this._charKeyFor("teacher", true);
    this.characters.set("teacher", teacherCh);

    // Students — fixed desk positions
    for (let i = 1; i < CHARACTER_PROFILES.length; i++) {
      const p = CHARACTER_PROFILES[i];
      const pos = desks[(i - 1) % desks.length];
      const ch = {
        x: pos.x, y: pos.y,
        targetX: pos.x, targetY: pos.y,
        facing: "down", frame: 0, frameCounter: 0,
        isTeacher: false,
      };
      ch.sprite = this._createCharSprite(p.id, false, pos.x, pos.y);
      ch.charKey = this._charKeyFor(p.id, false);
      this.characters.set(p.id, ch);
    }

    // Teacher focus indicator — a glowing circle that moves to target student
    this._teacherFocus = { x: teacherPos.x, y: teacherPos.y, targetX: teacherPos.x, targetY: teacherPos.y };
  }

  // ── Per-frame update: animate teacher focus indicator ───────────────────
  update() {
    this._frameCount++;

    // Smoothly move teacher focus indicator toward target
    if (this._teacherFocus) {
      const tf = this._teacherFocus;
      const dx = tf.targetX - tf.x;
      const dy = tf.targetY - tf.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const SPEED = 3;
      if (dist > SPEED) {
        tf.x += (dx / dist) * SPEED;
        tf.y += (dy / dist) * SPEED;
      } else {
        tf.x = tf.targetX;
        tf.y = tf.targetY;
      }
    }

    // Smoothly move character sprites toward their target positions and update animation state
    const CHAR_SPEED = 1.8;
    this.characters.forEach((ch) => {
      const dx = ch.targetX - ch.x;
      const dy = ch.targetY - ch.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const isWalking = dist > CHAR_SPEED;
      if (isWalking) {
        ch.x += (dx / dist) * CHAR_SPEED;
        ch.y += (dy / dist) * CHAR_SPEED;
        ch.facing = getFacing(dx, dy);
      }
      if (ch.sprite) {
        ch.sprite.x = ch.x;
        ch.sprite.y = ch.y;
        ch.sprite.setFlipX(ch.facing === "left");
        const key = ch.charKey;
        const desired = isWalking ? `${key}_walk` : `${key}_idle`;
        const cur = ch.sprite.anims && ch.sprite.anims.currentAnim;
        if ((!cur || cur.key !== desired) && this.anims.exists(desired)) {
          ch.sprite.play(desired);
        }
      }
    });

    // Draw overlay indicators
    this._drawAllCharacters();

    // Draw and fade interaction lines
    this._drawInteractionLines();
  }

  // ── Draw all characters ────────────────────────────────────────────────
  _drawAllCharacters() {
    if (!this.characterGraphics) return;
    this.characterGraphics.clear();
    const g = this.characterGraphics;

    // Destroy old labels
    if (this._nameLabels) {
      this._nameLabels.forEach((t) => t.destroy());
    }
    this._nameLabels = [];

    // ── Teacher focus indicator (animated glow that moves between students) ──
    if (this._teacherFocus) {
      const tf = this._teacherFocus;
      const pulse = 0.5 + 0.5 * Math.sin(this._frameCount * 0.06);

      // Outer soft glow
      g.fillStyle(0xfbbf24, 0.08 + 0.05 * pulse);
      g.fillCircle(tf.x, tf.y, 36);

      // Main ring
      g.lineStyle(2.5, 0xfbbf24, 0.7 + 0.3 * pulse);
      g.strokeCircle(tf.x, tf.y, 26);

      // Inner ring
      g.lineStyle(1, 0xffffff, 0.3 + 0.2 * pulse);
      g.strokeCircle(tf.x, tf.y, 22);

      // "선생님" label follows the focus
      const tLabel = this.add.text(tf.x, tf.y - 32, "선생님", {
        fontSize: "9px",
        fontFamily: "'Pretendard', sans-serif",
        color: "#fbbf24",
        stroke: "#000000",
        strokeThickness: 2,
        fontStyle: "bold",
      }).setOrigin(0.5, 1).setDepth(1001);
      this._nameLabels.push(tLabel);
    }

    // ── Fallback: draw graphics for characters whose sprite failed to load ──
    this.characters.forEach((ch) => {
      if (ch.sprite) return;
      const profile = CHARACTER_PROFILES.find((p) => {
        if (p.isTeacher && ch.isTeacher) return true;
        return !p.isTeacher && !ch.isTeacher;
      }) || CHARACTER_PROFILES[0];
      try {
        drawCharacter(g, ch.x, ch.y, profile.hair, profile.shirt, ch.facing, ch.frame, ch.isTeacher);
      } catch (e) {
        // ignore — fallback render failed
      }
    });

    // ── Student overlays (on top of background characters) ──
    this.characters.forEach((ch) => {
      if (ch.isTeacher) return; // teacher shown as focus indicator
      if (!ch._simData) return;

      const isTarget = ch._simId === this.targetStudentId;
      const statusCol = studentColor(ch._simData);

      // Status dot (small colored circle near the character)
      g.fillStyle(statusCol, 0.9);
      g.fillCircle(ch.x + 18, ch.y - 18, 4);

      // ADHD identified — blue underline
      if (ch._simData.is_identified) {
        g.lineStyle(2, 0x60a5fa, 0.85);
        g.lineBetween(ch.x - 16, ch.y + 22, ch.x + 16, ch.y + 22);
        // Blue badge dot
        g.fillStyle(0x60a5fa, 0.9);
        g.fillCircle(ch.x - 18, ch.y - 18, 4);
      }

      // Managed — purple corner marks
      if (ch._simData.is_managed) {
        g.lineStyle(2, 0xa855f7, 0.8);
        g.lineBetween(ch.x - 18, ch.y - 14, ch.x - 18, ch.y - 6);
        g.lineBetween(ch.x - 18, ch.y - 14, ch.x - 10, ch.y - 14);
        g.lineBetween(ch.x + 18, ch.y + 18, ch.x + 18, ch.y + 10);
        g.lineBetween(ch.x + 18, ch.y + 18, ch.x + 10, ch.y + 18);
      }

      // Name label above the background character
      const name = ch._simData?.name || ch._simId || "";
      if (name) {
        const color = isTarget ? "#fbbf24" : ch._simData.is_identified ? "#93c5fd" : "#e2e8f0";
        const label = this.add.text(ch.x, ch.y - 26, name, {
          fontSize: isTarget ? "10px" : "8px",
          fontFamily: "'Pretendard', sans-serif",
          color,
          stroke: "#000000",
          strokeThickness: 2,
          fontStyle: isTarget ? "bold" : "normal",
        }).setOrigin(0.5, 1).setDepth(1000);
        this._nameLabels.push(label);
      }
    });
  }

  // ── Draw playground below the reference image ──────────────────────────
  _drawPlayground() {
    const PY = 620;
    const PH = 130;
    const W = 800;

    // Use the provided pixel art soccer field image as playground background
    if (this.textures.exists("playground")) {
      if (this.playgroundImage) this.playgroundImage.destroy();
      this.playgroundImage = this.add.image(0, PY, "playground").setOrigin(0, 0);
      this.playgroundImage.setDisplaySize(W, PH);
      // Move below overlays but above background
      this.playgroundImage.setDepth(0.5);
    } else {
      // Fallback: simple green field
      const g = this.playgroundGraphics;
      g.fillStyle(0x4a8a3a);
      g.fillRect(0, PY, W, PH);
    }
  }

  // ── Area labels ────────────────────────────────────────────────────────
  _createAreaLabels() {
    const labelDefs = [
      { key: "classroom",  x: 16,  y: 16,  text: "교실" },
      { key: "office",     x: 540, y: 16,  text: "교무실" },
      { key: "hallway",    x: 16,  y: 448, text: "복도" },
      { key: "playground", x: 16,  y: 628, text: "운동장" },
    ];
    labelDefs.forEach(({ key, x, y, text }) => {
      const bg = this.add.graphics();
      bg.fillStyle(0x1e293b, 0.82);
      bg.fillRoundedRect(x - 4, y - 3, text.length * 11 + 14, 22, 4);
      bg.lineStyle(1, 0x334155, 0.7);
      bg.strokeRoundedRect(x - 4, y - 3, text.length * 11 + 14, 22, 4);
      const t = this.add.text(x + 3, y + 8, text, {
        fontSize: "11px",
        fontFamily: "'Pretendard', monospace",
        color: "#e2e8f0",
        fontStyle: "bold",
      }).setOrigin(0, 0.5);
      this.areaLabelTexts[key] = { bg, t };
    });
  }

  // ── Overlay + highlight ────────────────────────────────────────────────
  _drawOverlaysAndHighlight(activeArea) {
    if (!this.overlayGraphics) return;
    this.overlayGraphics.clear();
    this.highlightGraphics.clear();
    // No dim overlays, no borders — show the full map evenly
  }

  // ── Set active area with transition ────────────────────────────────────
  setActiveArea(area) {
    const prev = this.activeArea;
    this.activeArea = area;
    if (prev !== area) {
      const fromName = AREAS[prev]?.label || prev;
      const toName = AREAS[area]?.label || area;
      if (this.transitionLabel) {
        this.transitionLabel.setText(`${fromName} → ${toName}`).setAlpha(1);
        this.tweens.add({
          targets: this.transitionLabel,
          alpha: 0,
          duration: 1600,
          delay: 700,
          ease: "Power2",
        });
      }
    }
    this._drawOverlaysAndHighlight(area);
  }

  // ── Assign character targets based on area ─────────────────────────────
  _assignTargets(activeArea, targetId) {
    const areaPos = AREA_POSITIONS[activeArea];
    if (!areaPos) return;

    // Teacher target
    const teacherCh = this.characters.get("teacher");
    if (teacherCh) {
      const tPos = areaPos.teacher || AREA_POSITIONS.classroom.teacher;
      teacherCh.targetX = tPos.x;
      teacherCh.targetY = tPos.y;
    }

    // If teacher is observing a student, move toward them
    if (targetId && teacherCh) {
      const targetCh = this.characters.get(targetId);
      if (targetCh) {
        teacherCh.targetX = targetCh.targetX + 20;
        teacherCh.targetY = targetCh.targetY - 10;
      }
    }

    // Student targets
    const studentProfiles = CHARACTER_PROFILES.slice(1);
    studentProfiles.forEach((p, i) => {
      const ch = this.characters.get(p.id);
      if (!ch) return;

      // If this student goes to office for private_correction
      if (activeArea === "office" && p.id === targetId) {
        const officeSeats = AREA_POSITIONS.office.seats;
        ch.targetX = officeSeats[0].x;
        ch.targetY = officeSeats[0].y;
        return;
      }

      let positions;
      switch (activeArea) {
        case "classroom":
          positions = areaPos.desks;
          break;
        case "hallway":
          positions = areaPos.path;
          break;
        case "playground":
          positions = areaPos.scattered;
          break;
        default:
          positions = AREA_POSITIONS.classroom.desks;
      }

      const pos = positions[i % positions.length];
      ch.targetX = pos.x;
      ch.targetY = pos.y;
    });
  }

  // ── Map simulation student data to character profiles ──────────────────
  _mapStudentsToCharacters(simStudents) {
    // Assign simulation student ids to character profiles
    simStudents.forEach((student, idx) => {
      const profileIdx = idx + 1; // skip teacher
      if (profileIdx >= CHARACTER_PROFILES.length) return;
      const profileId = CHARACTER_PROFILES[profileIdx].id;
      const ch = this.characters.get(profileId);
      if (ch) {
        ch._simId = student.id;
        ch._simData = student;
      }
    });
  }

  // ── Interaction lines (v2) ──────────────────────────────────────────────
  showInteractionLines(interactions) {
    if (!interactions || !Array.isArray(interactions) || interactions.length === 0) return;
    const now = Date.now();

    interactions.forEach(({ actor, target, event_type }) => {
      // Find character profile ids that map to these simulation student ids
      let actorChId = null;
      let targetChId = null;
      this.characters.forEach((ch, id) => {
        if (ch._simId === actor) actorChId = id;
        if (ch._simId === target) targetChId = id;
      });
      if (!actorChId || !targetChId) return;

      const chA = this.characters.get(actorChId);
      const chB = this.characters.get(targetChId);
      if (!chA || !chB) return;

      // Color by interaction type
      let color = 0x94a3b8; // default gray
      if (event_type === "peer_conflict" || event_type === "peer_bullying") color = 0xef4444;
      else if (event_type === "peer_chat" || event_type === "peer_help" || event_type === "peer_friendship") color = 0x60a5fa;
      else if (event_type === "peer_contagion") color = 0xfbbf24;

      this._interactionLines.push({
        x1: chA.x, y1: chA.y,
        x2: chB.x, y2: chB.y,
        color,
        createdAt: now,
      });
    });
  }

  _drawInteractionLines() {
    if (!this.interactionGraphics) return;
    this.interactionGraphics.clear();
    const now = Date.now();
    const DURATION = 2000; // 2 seconds fade
    // Remove expired lines
    this._interactionLines = this._interactionLines.filter((l) => now - l.createdAt < DURATION);
    this._interactionLines.forEach((line) => {
      const elapsed = now - line.createdAt;
      const alpha = Math.max(0, 1 - elapsed / DURATION);
      this.interactionGraphics.lineStyle(2, line.color, alpha * 0.7);
      this.interactionGraphics.strokeLineShape(
        new Phaser.Geom.Line(line.x1, line.y1, line.x2, line.y2)
      );
      // Draw label at midpoint
      const mx = (line.x1 + line.x2) / 2;
      const my = (line.y1 + line.y2) / 2;
      this.interactionGraphics.fillStyle(line.color, alpha);
      this.interactionGraphics.fillCircle(mx, my, 3);
    });
  }

  // ── Bubble ─────────────────────────────────────────────────────────────
  showBubble(x, y, text, isTeacher) {
    this.bubbleGroup.clear(true, true);
    const maxWidth = 160;
    const padding = 6;
    const shortText = text.length > 60 ? text.substring(0, 57) + "..." : text;
    const bg = this.add.graphics();
    const textObj = this.add.text(x + padding, y + padding, shortText, {
      fontSize: "9px",
      fontFamily: "'Pretendard', sans-serif",
      color: isTeacher ? "#1e293b" : "#44403c",
      wordWrap: { width: maxWidth - padding * 2 },
      lineSpacing: 2,
    });
    const bounds = textObj.getBounds();
    const bw = Math.min(maxWidth, bounds.width + padding * 2 + 4);
    const bh = bounds.height + padding * 2 + 2;
    const bx = Math.max(4, Math.min(x, this.scale.width - bw - 4));
    const by = Math.max(4, Math.min(y, this.scale.height - bh - 8));
    textObj.setPosition(bx + padding + 2, by + padding);
    bg.fillStyle(isTeacher ? 0xffffff : 0xfef3c7, 0.95);
    bg.fillRoundedRect(bx, by, bw, bh, 4);
    bg.lineStyle(1, isTeacher ? 0x94a3b8 : 0xd97706, 0.5);
    bg.strokeRoundedRect(bx, by, bw, bh, 4);
    const tailX = Math.min(x + 12, bx + bw - 12);
    bg.fillStyle(isTeacher ? 0xffffff : 0xfef3c7, 0.95);
    bg.fillTriangle(tailX, by + bh, tailX + 6, by + bh, tailX + 3, by + bh + 6);
    this.bubbleGroup.add(bg);
    this.bubbleGroup.add(textObj);
    this.time.delayedCall(5000, () => { bg?.destroy(); textObj?.destroy(); });
  }

  // ── Classic mode update ────────────────────────────────────────────────
  updateSimState(state, events, scenario) {
    const lastEvent = events[events.length - 1];
    const turn = lastEvent?.time ?? 0;
    const action = lastEvent?.action || "";
    const area = deriveArea(scenario, action, turn, null);

    if (area !== this.activeArea) {
      this.setActiveArea(area);
    }

    // Move all characters to the active area positions
    this._assignTargets(area, null);

    if (state) {
      const risk = state.escalation_risk || 0;
      const compliance = state.compliance || 0;
      const distress = state.distress_level || 0;
      let statusLabel = "Calm", statusColor = "#60a5fa";
      if (risk > 0.7)           { statusLabel = "ESCALATING"; statusColor = "#ef4444"; }
      else if (distress > 0.6)  { statusLabel = "Distressed"; statusColor = "#f59e0b"; }
      else if (compliance > 0.7){ statusLabel = "Compliant"; statusColor = "#4ade80"; }
      if (this.statusText) {
        this.statusText.setText(
          `${statusLabel}  D:${(distress * 100).toFixed(0)}  C:${(compliance * 100).toFixed(0)}  A:${((state.attention || 0) * 100).toFixed(0)}  E:${(risk * 100).toFixed(0)}`
        );
        this.statusText.setColor(statusColor);
      }
    }

    if (lastEvent) {
      if (this.turnText && lastEvent.time !== undefined) {
        this.turnText.setText(`Turn ${lastEvent.time}`);
      }
      if (this.actionText && lastEvent.action) {
        const label = lastEvent.action.replace(/_/g, " ");
        this.actionText.setText(label);
        if (this.actionBg) {
          this.actionBg.clear();
          this.actionBg.fillStyle(0x1e293b, 0.8);
          const aw = label.length * 8 + 20;
          this.actionBg.fillRoundedRect(this.scale.width / 2 - aw / 2, 6, aw, 22, 4);
        }
      }
      if (lastEvent.utterance && lastEvent.speaker) {
        const teacherCh = this.characters.get("teacher");
        const bx = teacherCh ? teacherCh.x + 20 : 180;
        const by = teacherCh ? teacherCh.y - 30 : 60;
        this.showBubble(bx, by, `${lastEvent.speaker}: ${lastEvent.utterance}`, true);
      }
      if (lastEvent.student_narrative) {
        this.showBubble(200, 160, lastEvent.student_narrative, false);
      }
    }
  }

  // ── Multi + V2 mode update ──────────────────────────────────────────────
  updateMultiState(students, teacherAction, turn, classId, scenario, v2Location) {
    const actionType = teacherAction?.action_type || "";
    const targetId = teacherAction?.student_id || null;
    this.targetStudentId = targetId; // Store for indicator rendering

    // Map simulation students to character profiles (for name/state data)
    this._mapStudentsToCharacters(students);

    // Move teacher focus indicator to targeted student position
    if (targetId && this._teacherFocus) {
      // Find the character position for the target
      let targetPos = null;
      this.characters.forEach((ch) => {
        if (ch._simId === targetId) targetPos = { x: ch.x, y: ch.y };
      });
      if (targetPos) {
        this._teacherFocus.targetX = targetPos.x;
        this._teacherFocus.targetY = targetPos.y;
      }
    }

    if (this.turnText) {
      this.turnText.setText(`T${turn} C${classId}`);
    }

    const targetStudent = students.find((s) => s.id === targetId);
    const label = actionType
      ? `${actionType.replace(/_/g, " ")}${targetStudent ? ` → ${targetStudent.name}` : ""}`
      : "";
    if (this.actionText) {
      this.actionText.setText(label);
      if (this.actionBg && label) {
        this.actionBg.clear();
        this.actionBg.fillStyle(0x1e293b, 0.8);
        const aw = label.length * 7 + 20;
        this.actionBg.fillRoundedRect(this.scale.width / 2 - aw / 2, 6, aw, 22, 4);
      }
    }

    const managedCount = students.filter((s) => s.is_managed).length;
    const identifiedCount = students.filter((s) => s.is_identified).length;
    if (this.statusText) {
      this.statusText.setText(`식별: ${identifiedCount}  관리: ${managedCount}  학생: ${students.length}`);
      this.statusText.setColor("#94a3b8");
    }
  }

  // ── V2 mode update with interactions ──────────────────────────────────
  updateV2State(turnData) {
    const { students, teacher_action, turn, class_id, location, interactions } = turnData;
    if (!students) return;
    this.updateMultiState(students, teacher_action, turn, class_id, null, location || "classroom");
    if (interactions && Array.isArray(interactions) && interactions.length > 0) {
      this.showInteractionLines(interactions);
    }
  }
}

// ── React component ──────────────────────────────────────────────────────
export default function ClassroomView({ state, events, mode, multiTurnData, scenario, v2Info }) {
  const containerRef = useRef(null);
  const gameRef = useRef(null);
  const sceneRef = useRef(null);

  useEffect(() => {
    if (gameRef.current || !containerRef.current) return;

    const config = {
      type: Phaser.AUTO,
      parent: containerRef.current,
      width: 800,
      height: 750,
      backgroundColor: "#3b3558",
      pixelArt: true,
      scale: {
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH,
      },
      scene: UnifiedSchoolScene,
    };

    gameRef.current = new Phaser.Game(config);

    const checkScene = setInterval(() => {
      const scene = gameRef.current?.scene?.getScene("UnifiedSchoolScene");
      if (scene && scene.statusText) {
        sceneRef.current = scene;
        clearInterval(checkScene);
      }
    }, 100);

    return () => {
      clearInterval(checkScene);
      gameRef.current?.destroy(true);
      gameRef.current = null;
    };
  }, []);

  // Classic mode updates
  useEffect(() => {
    if (sceneRef.current && mode === "classic") {
      sceneRef.current.updateSimState(state, events ?? [], scenario);
    }
  }, [state, events, mode, scenario]);

  // Multi + V2 mode updates
  useEffect(() => {
    if (sceneRef.current && (mode === "multi" || mode === "v2") && multiTurnData) {
      if (mode === "v2") {
        sceneRef.current.updateV2State(multiTurnData);
      } else {
        const { students, teacher_action, turn, class_id } = multiTurnData;
        if (students) {
          sceneRef.current.updateMultiState(students, teacher_action, turn, class_id, scenario);
        }
      }
    }
  }, [multiTurnData, mode, scenario]);

  return (
    <div style={styles.container}>
      <div ref={containerRef} style={styles.canvas} />
    </div>
  );
}

const styles = {
  container: {
    flex: 1,
    position: "relative",
    borderRadius: 8,
    overflow: "hidden",
    border: "1px solid #334155",
    background: "#3b3558",
  },
  canvas: {
    width: "100%",
    height: "100%",
  },
};
