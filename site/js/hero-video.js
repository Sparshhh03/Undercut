// Intro background: plays the team-car clips one after another, forever.
// Every clip ends by diving into black, so the next clip simply fades up out of the black:
// a smooth dip-to-black between cars without any crossfaded double images.
//
// Two <video> elements take turns: while one plays, the other quietly loads the next clip.

const FADE_MS = 900;   // must match the opacity transition in styles.css
const LEAD_S = 0.35;   // start the next clip this long before the current one ends (it is black by then)

export function createHeroVideo(stage, clips, { reducedMotion = false, onChange } = {}) {
  const [a, b] = stage.querySelectorAll('video');
  // lighter files on phones, small screens and data-saver connections
  const small = matchMedia('(max-width: 900px)').matches || navigator.connection?.saveData;
  const src = (clip) => `${clip.base}-${small ? 720 : 1080}.mp4`;
  const poster = (clip) => `${clip.base}-poster.jpg`;

  // visitors who prefer less motion get a still of the first car
  stage.style.backgroundImage = `url("${poster(clips[0])}")`;
  if (reducedMotion || !a || !b) { onChange?.(clips[0], 0); return; }

  let index = 0, cur = a, next = b, switching = false, visible = true;

  const load = (video, clip) => {
    video.src = src(clip);
    video.poster = poster(clip);
    video.preload = 'auto';
    video.load();
  };

  const playCur = () => cur.play().catch(() => { /* autoplay blocked: the poster stays visible */ });

  function advance() {
    if (switching) return;
    switching = true;
    index = (index + 1) % clips.length;
    const incoming = next, outgoing = cur;
    incoming.currentTime = 0;
    incoming.classList.add('on');      // fades up out of black
    outgoing.classList.remove('on');   // already black at its end
    cur = incoming; next = outgoing;
    if (visible) playCur();
    onChange?.(clips[index], index);
    setTimeout(() => {
      outgoing.pause();
      if (clips.length > 1) load(outgoing, clips[(index + 1) % clips.length]);
      switching = false;
    }, FADE_MS + 100);
  }

  const onTime = (e) => {
    const v = e.target;
    if (v !== cur || switching || !v.duration) return;
    if (v.currentTime >= v.duration - LEAD_S) advance();
  };
  for (const v of [a, b]) {
    v.addEventListener('timeupdate', onTime);
    v.addEventListener('ended', (e) => { if (e.target === cur) advance(); });
  }

  load(cur, clips[0]);
  cur.classList.add('on');
  playCur();
  onChange?.(clips[0], 0);
  if (clips.length > 1) load(next, clips[1]);
  else cur.loop = true;

  // only play while the intro is on screen and the tab is visible (saves battery and data)
  const sync = () => (visible && !document.hidden ? playCur() : cur.pause());
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; sync(); }, { threshold: 0.05 }).observe(stage);
  document.addEventListener('visibilitychange', sync);
}
