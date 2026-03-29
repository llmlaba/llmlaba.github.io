import '../css/input.css';
import GLightbox from 'glightbox';
import 'glightbox/dist/css/glightbox.min.css';

document.addEventListener('DOMContentLoaded', () => {

  // ─── Gallery ─────────────────────────────────────────────────────────────
  GLightbox({ selector: '.glightbox' });

  // ─── Copy buttons for code blocks ────────────────────────────────────────
  document.querySelectorAll('pre code').forEach((block) => {
    const pre = block.parentNode;
    const btn = document.createElement('button');
    btn.textContent = 'Copy';
    btn.className = 'copy-btn';
    btn.setAttribute('aria-label', 'Copy code');
    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(block.textContent).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => (btn.textContent = 'Copy'), 2000);
      });
    });
    pre.appendChild(btn);
  });

  // ─── Sidebar: collapsible groups ─────────────────────────────────────────
  document.querySelectorAll('.sidebar-group-btn').forEach((btn) => {
    const items = btn.nextElementSibling;
    if (!items) return;

    // Auto-collapse groups that don't contain the active link
    const hasActive = items.querySelector('.sidebar-link.active');
    if (!hasActive) {
      items.classList.add('is-collapsed');
      btn.classList.add('is-collapsed');
    }

    btn.addEventListener('click', () => {
      const collapsed = items.classList.toggle('is-collapsed');
      btn.classList.toggle('is-collapsed', collapsed);
    });
  });

  // Scroll active sidebar link into view
  const activeLink = document.querySelector('#sidebar .sidebar-link.active');
  if (activeLink) {
    activeLink.scrollIntoView({ block: 'nearest' });
  }

  // ─── Mobile drawer ───────────────────────────────────────────────────────
  const menuBtn  = document.getElementById('mobile-menu-btn');
  const drawer   = document.getElementById('mobile-drawer');
  const overlay  = document.getElementById('mobile-overlay');

  function openDrawer() {
    drawer.classList.remove('hidden');
    overlay.classList.remove('hidden');
    menuBtn.setAttribute('aria-expanded', 'true');
  }

  function closeDrawer() {
    drawer.classList.add('hidden');
    overlay.classList.add('hidden');
    menuBtn.setAttribute('aria-expanded', 'false');
  }

  menuBtn?.addEventListener('click', () => {
    drawer.classList.contains('hidden') ? openDrawer() : closeDrawer();
  });

  overlay?.addEventListener('click', closeDrawer);

  // Also collapse groups in the mobile drawer
  drawer?.querySelectorAll('.sidebar-group-btn').forEach((btn) => {
    const items = btn.nextElementSibling;
    if (!items) return;
    const hasActive = items.querySelector('.sidebar-link.active');
    if (!hasActive) {
      items.classList.add('is-collapsed');
      btn.classList.add('is-collapsed');
    }
    btn.addEventListener('click', () => {
      const collapsed = items.classList.toggle('is-collapsed');
      btn.classList.toggle('is-collapsed', collapsed);
    });
  });

  // ─── Tabs ─────────────────────────────────────────────────────────────────
  initTabs();

  // ─── Home: category filter ────────────────────────────────────────────────
  initHomeFilter();

});

// ─── Home filter ─────────────────────────────────────────────────────────────

const CAT_LABELS = {
  all:      'All Articles',
  llm:      'LLM / Software',
  software: 'LLM / Software',
  hardware: 'GPU / Hardware',
  gpu:      'GPU / Hardware',
  general:  'General Reference',
  project:  'Projects',
  issue:    'Known Issues',
  fail:     'Failed Tests',
};

function initHomeFilter() {
  const links = document.querySelectorAll('.home-cat-link');
  if (links.length === 0) return;

  // Restore category from URL hash
  const hash = window.location.hash.replace('#', '');
  if (hash && hash !== 'all') applyFilter(hash);

  links.forEach((link) => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      applyFilter(link.dataset.cat);
      // Close mobile drawer if open
      document.getElementById('mobile-drawer')?.classList.add('hidden');
      document.getElementById('mobile-overlay')?.classList.add('hidden');
    });
  });
}

function applyFilter(cat) {
  // Show/hide cards
  document.querySelectorAll('.article-card').forEach((card) => {
    const cats = card.dataset.categories.split(' ');
    const visible = cat === 'all' || cats.includes(cat);
    card.style.display = visible ? '' : 'none';
  });

  // Update active link (both desktop sidebar and mobile drawer)
  document.querySelectorAll('.home-cat-link').forEach((link) => {
    link.classList.toggle('active', link.dataset.cat === cat);
  });

  // Update title
  const titleEl = document.getElementById('home-category-title');
  if (titleEl) titleEl.textContent = CAT_LABELS[cat] || cat;

  // Update URL hash
  history.replaceState(null, '', cat === 'all' ? location.pathname : '#' + cat);
}

// ─── Tab logic ───────────────────────────────────────────────────────────────

function showTab(tabId) {
  document.querySelectorAll('[data-tab-section]').forEach((el) => {
    el.style.display = el.dataset.tabSection === tabId ? '' : 'none';
  });
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  history.replaceState(null, '', '#' + tabId);
}

function initTabs() {
  const article = document.querySelector('#article-content article');
  if (!article) return;

  // ── Hide duplicate title and date from article content ──
  // (they are already rendered in the layout header)
  const firstH1 = article.querySelector('h1');
  if (firstH1) firstH1.hidden = true;

  // Hide > Date: blockquote (first blockquote that starts with "Date")
  const firstBQ = article.querySelector('blockquote');
  if (firstBQ && /^date/i.test(firstBQ.textContent.trim())) {
    firstBQ.hidden = true;
  }

  // ── Find ToC ──
  const tocH2 = article.querySelector('h2#table-of-contents');
  if (!tocH2) return; // article has no ToC — nothing to do

  const tocUl = tocH2.nextElementSibling;
  if (!tocUl || tocUl.tagName !== 'UL') return;

  // Extract tab definitions: [{label, id}]
  const tabs = [...tocUl.querySelectorAll('a')].map((a) => ({
    label: a.textContent.trim(),
    id:    a.getAttribute('href').replace('#', ''),
  }));

  if (tabs.length === 0) return;

  // Hide ToC section itself
  tocH2.hidden = true;
  tocUl.hidden = true;

  // IDs that mark tab boundaries
  const tabIds = new Set(tabs.map((t) => t.id));

  // Walk article children and tag each with its section
  let currentSection = null;
  for (const el of article.children) {
    if (el.hidden) continue;

    if (el.tagName === 'H2' && tabIds.has(el.id)) {
      currentSection = el.id;
      el.hidden = true; // the H2 becomes the tab button label
      continue;
    }

    if (currentSection) {
      el.dataset.tabSection = currentSection;
    }
  }

  // Single tab → skip UI, just show everything
  if (tabs.length <= 1) return;

  // Build tab buttons
  const tabNav = document.getElementById('tab-nav');
  if (!tabNav) return;

  tabs.forEach((tab) => {
    const btn = document.createElement('button');
    btn.textContent = tab.label;
    btn.className   = 'tab-btn';
    btn.dataset.tab = tab.id;
    btn.addEventListener('click', () => showTab(tab.id));
    tabNav.appendChild(btn);
  });

  // Activate initial tab (from URL hash, or first tab)
  const hash    = window.location.hash.replace('#', '');
  const initial = tabs.find((t) => t.id === hash) ? hash : tabs[0].id;
  showTab(initial);
}
