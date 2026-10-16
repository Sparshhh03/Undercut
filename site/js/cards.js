// Undercut: driver cards. Each driver's season as a trading card (ratings computed in
// f1predict/ratings.py), with a stylised helmet drawn in team colours, and a "walkout" reveal
// when a card is opened: nationality, team, car number, then the card flips in.

// Helmet colours per team: [shell, accent, trim]
const HELMET = {
  mercedes: ['#1A1D22', '#19C3AC', '#C9CED4'], ferrari: ['#E8002D', '#FFD100', '#FFFFFF'],
  mclaren: ['#FF8000', '#14161A', '#4FC3F7'], red_bull: ['#1B2A5C', '#FFC72C', '#E4002B'],
  rb: ['#F4F5F8', '#1E41FF', '#E4002B'], alpine: ['#0A2C8C', '#F282B4', '#FFFFFF'],
  williams: ['#0A1F44', '#64C4FF', '#FFFFFF'], aston_martin: ['#0B5B43', '#CEDC00', '#FFFFFF'],
  haas: ['#F4F5F8', '#E6002D', '#1A1D22'], audi: ['#9EA4AA', '#E30613', '#1A1D22'],
  cadillac: ['#16181C', '#C9CDD1', '#C8A45C'],
};
const CODE = {
  mercedes: 'MER', ferrari: 'FER', mclaren: 'MCL', red_bull: 'RBR', rb: 'RB', alpine: 'ALP',
  williams: 'WIL', aston_martin: 'AMR', haas: 'HAA', audi: 'AUD', cadillac: 'CAD',
};
const STAT_INFO = {
  QUA: ['Qualifying', r => `Average qualifying position ${r.avg_quali}`],
  PAC: ['Pace', r => `Average finish ${r.avg_finish} when reaching the flag`],
  RAC: ['Racecraft', r => `${r.places_gained >= 0 ? '+' : ''}${r.places_gained} places from the grid per finish`],
  CON: ['Consistency', r => `Finished ${Math.round(r.finish_rate * 100)}% of races`],
  FRM: ['Form', r => `${r.points_last5} points per race over the last five`],
  H2H: ['Vs teammate', r => `Beat the teammate in ${Math.round(r.h2h * 100)}% of head-to-heads`],
};

// Real helmet photos (front view, background removed) for drivers who have one; the rest get the drawn helmet.
const HELMET_PHOTOS = {
  max_verstappen: 'media/helmets/verstappen.webp', antonelli: 'media/helmets/antonelli.webp',
  russell: 'media/helmets/russell.webp', leclerc: 'media/helmets/leclerc.webp', hamilton: 'media/helmets/hamilton.webp',
};
// Driver photos used as the card's background (the helmet then moves under the rating as a badge)
const CARD_BG = {
  max_verstappen: 'media/drivers/verstappen.webp', leclerc: 'media/drivers/leclerc.webp', hamilton: 'media/drivers/hamilton.webp',
  russell: 'media/drivers/russell.webp', antonelli: 'media/drivers/antonelli.webp',
  norris: 'media/drivers/norris.webp', piastri: 'media/drivers/piastri.webp',
};

const hash = s => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7);

function helmetSVG(d, teamId) {
  const [shell, accent, trim] = HELMET[teamId] || ['#3A3F47', '#8C8C8C', '#FFFFFF'];
  const id = 'h-' + d.driver;
  const style = hash(d.driver) % 3;
  const pattern = [
    // swoosh over the crown
    `<path d="M0 74 C60 38 132 30 210 62 L210 92 C140 62 70 66 0 104 Z" fill="${accent}"/>
     <path d="M0 104 C70 66 140 62 210 92 L210 98 C140 69 70 73 0 110 Z" fill="${trim}"/>`,
    // lower split
    `<path d="M0 128 L210 92 L210 180 L0 180 Z" fill="${accent}"/>
     <path d="M0 122 L210 86 L210 92 L0 128 Z" fill="${trim}"/>`,
    // cap
    `<path d="M0 0 L210 0 L210 60 C150 46 86 46 0 72 Z" fill="${accent}"/>
     <path d="M0 72 C86 46 150 46 210 60 L210 66 C150 53 86 53 0 78 Z" fill="${trim}"/>`,
  ][style];
  return `<svg class="helmet" viewBox="0 0 210 180" aria-hidden="true">
    <defs>
      <clipPath id="${id}-c"><path d="M20 120 C14 100 20 78 34 62 C54 38 88 24 124 24 C162 24 190 46 196 80 C200 102 194 122 182 136 L174 144 C146 149 108 152 70 152 C46 152 28 140 20 120 Z"/></clipPath>
      <linearGradient id="${id}-v" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${accent}" stop-opacity=".55"/><stop offset=".45" stop-color="#0B0E15"/><stop offset="1" stop-color="#05070B"/></linearGradient>
      <radialGradient id="${id}-s" cx=".32" cy=".22" r=".85"><stop offset="0" stop-color="#fff" stop-opacity=".38"/><stop offset=".4" stop-color="#fff" stop-opacity=".04"/><stop offset="1" stop-color="#000" stop-opacity=".5"/></radialGradient>
    </defs>
    <g clip-path="url(#${id}-c)">
      <rect width="210" height="180" fill="${shell}"/>
      ${pattern}
      <text x="156" y="128" text-anchor="middle" class="h-num" fill="${trim}">${d.number || ''}</text>
      <rect width="210" height="180" fill="url(#${id}-s)"/>
    </g>
    <path d="M146 27 L192 41 L190 51 L144 37 Z" fill="${trim}" opacity=".9"/>
    <path d="M26 88 C38 71 66 62 104 63 C121 64 130 72 128 83 L125 95 C123 102 116 105 107 105 L36 105 C28 102 24 95 26 88 Z" fill="url(#${id}-v)" stroke="${trim}" stroke-opacity=".55" stroke-width="2.5"/>
    <path d="M38 79 C58 70 82 67 106 68" fill="none" stroke="#fff" stroke-opacity=".35" stroke-width="3" stroke-linecap="round"/>
    <path d="M30 132 C52 146 100 148 176 140 L174 146 C146 151 108 154 70 154 C48 154 36 146 30 132 Z" fill="#000" opacity=".5"/>
  </svg>`;
}

function cardHTML(d, team, tag = 'button') {
  const c = d.card;
  const [teamName, teamColor] = team(d.team);
  const surname = d.name.split(' ').slice(-1)[0];
  const stats = Object.entries(c.stats).map(([k, v]) => `<span class="dc-stat"><b>${v}</b><i>${k}</i></span>`).join('');
  const attrs = tag === 'button'
    ? `type="button" data-driver="${d.driver}" aria-label="${d.name}, ${teamName}, rated ${c.ovr}. Open card"`
    : 'aria-hidden="true"';
  const bg = CARD_BG[d.driver];
  return `<${tag} class="dcard ${c.tier}${bg ? ' has-bg' : ''}" style="--team:${teamColor}" ${attrs}>
    <span class="dc-frame"><span class="dc-face">
      ${bg ? `<span class="dc-bgimg" style="background-image:url('${bg}')"></span>` : ''}
      <span class="dc-band"></span>
      <span class="dc-top">
        <span class="dc-ovr">${c.ovr}</span>
        <span class="dc-team">${CODE[d.team] || d.team.slice(0, 3).toUpperCase()}</span>
        <span class="dc-no">${d.number || ''}</span>
      </span>
      ${HELMET_PHOTOS[d.driver]
        ? `<span class="dc-helmet photo"><img src="${HELMET_PHOTOS[d.driver]}" alt="" decoding="async"></span>`
        : `<span class="dc-helmet">${helmetSVG(d, d.team)}</span>`}
      <span class="dc-name">${surname}</span>
      <span class="dc-stats">${stats}</span>
    </span></span>
  </${tag}>`;
}

export function initCards({ data, team, finChip, short, esc, G, RM }) {
  const S = data.standings;
  const drivers = S.drivers.filter(d => d.card);
  const grid = document.querySelector('#card-grid');
  const sortBtns = document.querySelectorAll('#card-sort button');
  let order = 'rating';

  function render() {
    const list = [...drivers].sort(order === 'rating' ? (a, b) => b.card.ovr - a.card.ovr || a.pos - b.pos : (a, b) => a.pos - b.pos);
    grid.innerHTML = list.map(d => cardHTML(d, team)).join('');
  }
  sortBtns.forEach(b => b.addEventListener('click', () => {
    order = b.dataset.sort;
    sortBtns.forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    render();
  }));
  render();

  // foil sheen follows the pointer on the card under it
  grid.addEventListener('pointermove', e => {
    const c = e.target.closest('.dcard');
    if (!c) return;
    const r = c.getBoundingClientRect();
    c.style.setProperty('--sx', ((e.clientX - r.left) / r.width * 100) + '%');
    c.style.setProperty('--sy', ((e.clientY - r.top) / r.height * 100) + '%');
  }, { passive: true });

  // ---------- reveal ----------
  const dlg = document.querySelector('#card-reveal');
  let tl = null, shown = null;

  function panelHTML(d) {
    const c = d.card;
    const [teamName] = team(d.team);
    const by = new Map(d.finishes.map(f => [f.round, f]));
    const rows = Object.entries(c.stats).map(([k, v]) => `
      <div class="rv-stat">
        <span class="rv-k">${STAT_INFO[k][0]}</span>
        <span class="rv-bar"><i style="--v:${v}%"></i></span>
        <b class="rv-v">${v}</b>
        <span class="rv-why">${esc(STAT_INFO[k][1](c.raw))}</span>
      </div>`).join('');
    return `
      <h3 id="rv-title" class="rv-name">${esc(d.name)}</h3>
      <p class="rv-sub">${esc(teamName)}, car ${esc(d.number || '')}. ${esc(d.nationality)}.</p>
      <dl class="rv-facts">
        <div><dt>Championship</dt><dd>P${d.pos}</dd></div>
        <div><dt>Points</dt><dd>${d.points}</dd></div>
        <div><dt>Wins</dt><dd>${d.wins}</dd></div>
        <div><dt>Podiums</dt><dd>${d.podiums}</dd></div>
        <div><dt>Poles</dt><dd>${d.poles}</dd></div>
      </dl>
      <div class="rv-stats">${rows}</div>
      <p class="rv-season-h">Every Grand Prix this season</p>
      <span class="strip">${S.rounds.map(r => finChip(by.get(r.round), `R${r.round} ${short(r.name)}`)).join('')}</span>`;
  }

  function fill(d) {
    shown = d;
    const [teamName, teamColor] = team(d.team);
    dlg.style.setProperty('--team', teamColor);
    dlg.querySelector('.rv-card').innerHTML = cardHTML(d, team, 'div');
    dlg.querySelector('.rv-panel').innerHTML = panelHTML(d);
    const beats = dlg.querySelectorAll('.rv-beat');
    beats[0].textContent = d.nationality;
    beats[1].textContent = teamName;
    beats[2].textContent = d.number ? '#' + d.number : '';
    const i = drivers.indexOf(d);
    dlg.querySelector('.rv-prev').disabled = i <= 0;
    dlg.querySelector('.rv-next').disabled = i >= drivers.length - 1;
  }

  let watchdog;
  function finish() {
    clearTimeout(watchdog);
    if (tl) { tl.progress(1); tl.kill(); tl = null; }
    dlg.classList.remove('walking');
  }

  function open(d) {
    fill(d);
    if (!dlg.open) dlg.showModal();
    dlg.querySelector('.rv-close').focus();
    if (!G || RM) { dlg.classList.remove('walking'); return; }
    dlg.classList.add('walking');
    const card = dlg.querySelector('.rv-card .dcard');
    const beats = dlg.querySelectorAll('.rv-beat');
    tl = G.timeline({ onComplete: finish });
    watchdog = setTimeout(finish, 5000);  // if the browser stalls animation frames, still end on the card
    tl.fromTo(dlg.querySelector('.rv-beam'), { scaleY: 0, opacity: 0 }, { scaleY: 1, opacity: 1, duration: 0.5, ease: 'power3.out' });
    beats.forEach((b, k) => {
      if (!b.textContent) return;
      tl.fromTo(b, { opacity: 0, y: 24, letterSpacing: '.3em' }, { opacity: 1, y: 0, letterSpacing: '0em', duration: 0.3, ease: 'power3.out' }, k ? '>' : '-=0.15')
        .to(b, { opacity: 0, y: -16, duration: 0.18, ease: 'power2.in' }, '+=0.3');
    });
    tl.fromTo(card, { rotateY: 88, scale: 0.55, y: 60, opacity: 0 }, { rotateY: 0, scale: 1, y: 0, opacity: 1, duration: 0.7, ease: 'power3.out' })
      .fromTo(dlg.querySelector('.rv-flash'), { opacity: 0.85 }, { opacity: 0, duration: 0.6, ease: 'power2.out', immediateRender: false }, '-=0.35')
      .fromTo(dlg.querySelector('.rv-panel'), { opacity: 0, x: 24 }, { opacity: 1, x: 0, duration: 0.45, ease: 'power3.out' }, '-=0.3')
      .fromTo(dlg.querySelectorAll('.rv-bar i'), { scaleX: 0 }, { scaleX: 1, duration: 0.6, stagger: 0.05, ease: 'power3.out' }, '-=0.3');
  }

  function step(dir) {
    const i = drivers.indexOf(shown) + dir;
    if (i < 0 || i >= drivers.length) return;
    finish();
    fill(drivers[i]);
    if (G && !RM) G.fromTo(dlg.querySelector('.rv-card .dcard'), { opacity: 0, x: 30 * dir }, { opacity: 1, x: 0, duration: 0.35, ease: 'power3.out' });
  }

  grid.addEventListener('click', e => {
    const c = e.target.closest('.dcard');
    if (c) open(drivers.find(d => d.driver === c.dataset.driver));
  });
  dlg.addEventListener('click', e => {
    if (dlg.classList.contains('walking')) { finish(); return; }   // a click skips the walkout
    if (e.target === dlg) dlg.close();                               // click outside the content
  });
  dlg.querySelector('.rv-close').addEventListener('click', e => { e.stopPropagation(); dlg.close(); });
  dlg.querySelector('.rv-prev').addEventListener('click', e => { e.stopPropagation(); step(-1); });
  dlg.querySelector('.rv-next').addEventListener('click', e => { e.stopPropagation(); step(1); });
  dlg.addEventListener('keydown', e => {
    if (dlg.classList.contains('walking') && e.key !== 'Escape') { finish(); e.preventDefault(); return; }
    if (e.key === 'ArrowRight') step(1);
    if (e.key === 'ArrowLeft') step(-1);
  });
  dlg.addEventListener('close', () => {
    finish();
    grid.querySelector(`[data-driver="${shown?.driver}"]`)?.focus();
  });
}
