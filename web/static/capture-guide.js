/* Camera orientation is a reference only; server geometry remains authoritative. */
(function (root) {
  'use strict';
  // Right foot viewed from above: big toe on the viewer's LEFT.
  const rightFootPath = 'M141 326 C121 326 116 310 121 286 C132 257 139 232 132 210 C115 181 108 152 105 121 C100 103 100 79 114 74 C126 68 135 73 137 85 L139 105 C139 78 158 78 159 90 L161 112 C164 92 180 96 180 108 L180 130 C185 115 198 123 197 137 L196 161 C205 146 215 160 211 177 C207 201 197 211 191 238 L180 295 C178 316 165 330 141 326 Z';
  function guideTransform(foot, width = 300) {
    return foot === 'left' ? `translate(${width} 0) scale(-1 1)` : '';
  }
  function sideName(foot, key) {
    if (key === 'image_measure') return '正上方';
    const medial = (foot === 'right') === (key === 'image_left');
    return (key === 'image_left' ? '画面左侧' : '画面右侧') + ' · ' + (medial ? '足内侧（大拇趾侧）' : '足外侧（小拇趾侧）');
  }
  function tilt(beta, gamma) {
    if (!Number.isFinite(beta) || !Number.isFinite(gamma)) return null;
    // Device +z points out of the screen. Rear camera points toward -z.
    const z = Math.cos(beta * Math.PI / 180) * Math.cos(gamma * Math.PI / 180);
    if (z <= 0) return null; // Rear camera must face down, not up.
    return Math.acos(Math.min(1, z)) * 180 / Math.PI;
  }
  function createAngleGate(top) {
    let since = null, last = null, value = null;
    return {
      sample(angle, now) {
        const good = Number.isFinite(angle) && (top ? angle >= 0 && angle <= 5 : angle >= 30 && angle <= 35);
        if (!good || last === null || now - last > 350) since = null;
        if (good && since === null) since = now;
        value = good ? angle : null; last = now;
      },
      ready(now) { return since !== null && value !== null && now - since >= 500 && now - last <= 350; },
      value() { return value; }
    };
  }
  if (typeof module !== 'undefined') module.exports = {sideName, tilt, guideTransform, createAngleGate};
  if (!root.document) return;
  let panel, stream, timer, gateTimer, session = 0, onOrientation, returnFocus, oldOverflow;
  const $ = id => document.getElementById(id);
  function stopCamera() {
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null;
    if ($('capture-video')) $('capture-video').srcObject = null;
  }
  function close() {
    session++;
    stopCamera();
    clearInterval(timer);
    clearInterval(gateTimer);
    window.removeEventListener('deviceorientation', onOrientation);
    if (panel && !panel.hidden) {
      panel.hidden = true;
      document.body.style.overflow = oldOverflow;
      if (returnFocus) returnFocus.focus();
    }
  }
  function init() {
    panel = document.createElement('section');
    panel.className = 'capture-dialog'; panel.hidden = true;
    panel.setAttribute('role', 'dialog'); panel.setAttribute('aria-modal', 'true');
    panel.setAttribute('aria-labelledby', 'capture-title');
    panel.innerHTML = `<div class="capture-body">
      <div class="capture-header"><h2 id="capture-title"></h2><button id="capture-close" type="button">关闭</button></div>
      <div class="capture-guide-text" id="capture-instruction"></div>
      <div class="capture-position"><svg viewBox="0 0 160 120" aria-label="俯视拍摄位置示意">
        <rect x="51" y="7" width="58" height="94" rx="3" fill="white"/>
        <g id="capture-mini-foot"><path d="${rightFootPath}" transform="translate(17 -14) scale(.4 .34)" fill="#9bd2c6"/></g>
        <text x="80" y="116" fill="white" text-anchor="middle" font-size="10">脚跟朝自己 · 脚尖朝前</text>
        <circle id="capture-camera-dot" cx="80" cy="50" r="10" fill="#ec7859"/>
        <path id="capture-camera-line" d="M80 50L80 50" stroke="#ec7859" stroke-width="3" stroke-dasharray="4 3"/>
      </svg><div class="capture-guide-text" id="capture-position-text"></div></div>
      <div class="capture-stage"><video id="capture-video" autoplay muted playsinline></video>
        <svg class="capture-outline" viewBox="0 0 300 400" preserveAspectRatio="xMidYMid meet" aria-label="脚形和纸张参考轮廓">
          <rect x="57" y="45" width="186" height="290" rx="3" fill="none" stroke="white" stroke-width="2" stroke-dasharray="9 7"/>
          <path id="capture-foot-outline" d="${rightFootPath}" fill="none" stroke="#73ffca" stroke-width="3" stroke-dasharray="8 5"/>
          <text id="capture-big-toe" x="113" y="63" fill="#73ffca" text-anchor="middle" font-size="12">大拇趾</text>
          <text x="150" y="362" fill="white" text-anchor="middle" font-size="12">轮廓仅作构图参考，不必完全重合</text>
        </svg></div>
      <div id="capture-angle" class="capture-angle" data-state="unknown">角度参考未开启</div>
      <div id="capture-status" class="capture-warning" role="status"></div>
      <div class="capture-actions"><button id="capture-start" type="button">开启相机</button><button id="capture-sensor" type="button">开启角度参考</button><button id="capture-shoot" type="button" class="capture-primary" disabled>拍下并使用</button><button id="capture-native" type="button">使用系统相机</button></div>
      <p class="capture-note">保持三张照片都竖拍、使用同一后置镜头，不变焦。纸张平整、全程不移动脚。虚线不会写入照片。</p>
      <p class="capture-note">倾角来自手机传感器，不是照片几何检测：无法确认手机是否已移到正确侧面。系统相机内无法显示本页叠加引导；上传后仍需通过测量检查。</p>
    </div>`;
    document.body.appendChild(panel);
    $('capture-close').onclick = close;
    panel.addEventListener('keydown', e => {
      if (e.key === 'Escape') close();
      if (e.key === 'Tab') {
        const buttons = [...panel.querySelectorAll('button:not(:disabled)')];
        const first = buttons[0], last = buttons[buttons.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    });
    window.addEventListener('pagehide', close);
    document.addEventListener('visibilitychange', () => { if (document.hidden) close(); });
  }
  function open(key, accept, nativeInput) {
    const foot = document.querySelector('input[name="footSide"]:checked')?.value;
    if (!foot) { alert('请先选择左脚或右脚，再开始拍照。'); return; }
    if (!panel) init();
    close();
    const token = session;
    returnFocus = document.activeElement; oldOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden'; panel.hidden = false;
    const top = key === 'image_measure', left = key === 'image_left';
    const gate = createAngleGate(top);
    let cameraReady = false, capturing = false;
    const updateShutter = () => {
      const ready = gate.ready(performance.now());
      $('capture-shoot').disabled = !cameraReady || capturing || !ready;
      $('capture-shoot').textContent = capturing ? '正在生成照片…' : ready ? '拍下并使用' : '角度稳定达标后可拍';
    };
    gateTimer = setInterval(updateShutter, 100);
    $('capture-title').textContent = (foot === 'right' ? '右脚 · ' : '左脚 · ') + sideName(foot, key);
    $('capture-instruction').textContent = top ? '手机放在脚的正上方，后置镜头朝下，完整拍入脚和 A4 纸，脚跟贴齐纸张底部短边。' : '从正上方把手机移到脚的' + (left ? '左' : '右') + '侧，再让镜头对准脚中心；不要只在原地转动手机。推荐偏离正上方 30°–35°。';
    $('capture-position-text').textContent = top ? '橙点：手机在脚正上方。' : '橙点：手机应移动到这一侧。另一张斜拍必须从相反侧拍摄。';
    $('capture-camera-dot').setAttribute('cx', top ? 80 : left ? 22 : 138);
    $('capture-camera-line').setAttribute('d', top ? 'M80 50L80 50' : `M${left ? 22 : 138} 50L80 50`);
    $('capture-foot-outline').setAttribute('transform', guideTransform(foot));
    $('capture-mini-foot').setAttribute('transform', guideTransform(foot, 160));
    $('capture-big-toe').setAttribute('x', foot === 'left' ? '187' : '113');
    $('capture-angle').textContent = '角度参考未开启'; $('capture-angle').dataset.state = 'unknown';
    $('capture-sensor').disabled = false; $('capture-start').disabled = false; $('capture-shoot').disabled = true;
    $('capture-status').textContent = '开启相机及角度参考，角度连续达标半秒后才可拍摄。无法读取角度可用系统相机，随后进行照片质检。';
    $('capture-close').focus();
    $('capture-native').onclick = () => { close(); nativeInput.value = ''; nativeInput.click(); };
    $('capture-start').onclick = async () => {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        $('capture-status').textContent = '当前环境无法使用网页取景。请使用 HTTPS 地址，或点击“使用系统相机”。'; return;
      }
      $('capture-start').disabled = true;
      $('capture-status').textContent = '正在请求后置相机权限…';
      try {
        const acquired = await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},width:{ideal:1920},height:{ideal:1440}}});
        if (token !== session) { acquired.getTracks().forEach(t => t.stop()); return; }
        stream = acquired;
        const video = $('capture-video'); video.srcObject = stream; await video.play();
        if (token !== session) return;
        const facing = stream.getVideoTracks()[0].getSettings().facingMode;
        $('capture-status').textContent = facing === 'user' ? '当前使用前置镜头，请改用系统相机选择后置镜头。' : '请开启角度参考，并确认镜头对准脚中心。避免侧面强光在纸上投出大片阴影；拍后会逐张质检。';
        cameraReady = facing !== 'user';
        stream.getVideoTracks()[0].addEventListener('ended', () => { if (token === session) { cameraReady = false; updateShutter(); } });
        updateShutter();
      } catch (_) {
        if (token !== session) return;
        stopCamera(); $('capture-start').disabled = false;
        $('capture-status').textContent = '相机未能打开：请检查权限，或使用系统相机/选择文件。';
      }
    };
    $('capture-sensor').onclick = async () => {
      const Orientation = window.DeviceOrientationEvent;
      if (!window.isSecureContext || !Orientation) { $('capture-angle').textContent = '无法读取角度，网页快门保持锁定。可使用系统相机，再进行照片质检。'; return; }
      $('capture-sensor').disabled = true;
      try {
        if (typeof Orientation.requestPermission === 'function' && await Orientation.requestPermission() !== 'granted') throw new Error('denied');
        if (token !== session) return;
        $('capture-sensor').disabled = true;
        let last = 0;
        $('capture-angle').textContent = '正在等待传感器…';
        onOrientation = e => {
          const angle = tilt(e.beta, e.gamma);
          gate.sample(angle, performance.now()); updateShutter();
          last = Date.now();
          const el = $('capture-angle');
          if (angle === null) { el.textContent = '请让后置镜头朝向地面；当前读数不可用。'; el.dataset.state = 'unknown'; return; }
          const good = top ? angle <= 5 : angle >= 30 && angle <= 35;
          el.dataset.state = good ? 'good' : 'adjust';
          el.textContent = `倾角参考 ${angle.toFixed(1)}° · ` + (good ? '进入参考区间，请确认位置' : top ? '正上方参考 0°–5°' : angle < 30 ? '向脚侧面移动并对准脚中心' : '向正上方靠近并对准脚中心');
        };
        window.addEventListener('deviceorientation', onOrientation);
        timer = setInterval(() => {
          if (Date.now() - last > 1000) { $('capture-angle').textContent = '传感器未更新，快门已锁定；可使用系统相机并接受照片质检。'; $('capture-angle').dataset.state = 'unknown'; }
        }, 1000);
      } catch (_) { if (token === session) { $('capture-angle').textContent = '未取得角度权限，网页快门保持锁定；可使用系统相机并接受照片质检。'; $('capture-sensor').disabled = false; } }
    };
    $('capture-shoot').onclick = () => {
      const video = $('capture-video');
      if (!video.videoWidth || !video.videoHeight || !stream || !cameraReady || capturing || !gate.ready(performance.now())) { updateShutter(); return; }
      if (video.videoWidth > video.videoHeight) { $('capture-status').textContent = '请把手机竖起来再拍摄，以保持三张照片方向一致。'; return; }
      const captureAngle = gate.value();
      capturing = true; updateShutter();
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      canvas.toBlob(blob => {
        if (token !== session) return;
        if (!blob) { $('capture-status').textContent = '照片生成失败，请重试。'; capturing = false; updateShutter(); return; }
        accept(key, new File([blob], key + '.jpg', {type:'image/jpeg'}), {angle:captureAngle}); close();
      }, 'image/jpeg', .95);
    };
  }
  function updateLabels() {
    const foot = document.querySelector('input[name="footSide"]:checked')?.value;
    for (const key of ['image_left','image_right']) {
      const label = document.querySelector(`#slot-${key} .slot-label`);
      if (label?.firstChild) label.firstChild.textContent = foot ? sideName(foot,key) : (key === 'image_left' ? '左侧斜拍' : '右侧斜拍');
    }
  }
  document.querySelectorAll('input[name="footSide"]').forEach(el => el.addEventListener('change', updateLabels));
  updateLabels();
  root.FootCapture = {open, close};
})(typeof window === 'undefined' ? globalThis : window);
