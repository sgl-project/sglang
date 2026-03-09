/**
 * Add "Copy" option inside the download dropdown (.md, .pdf, Copy).
 * Copies the current page's Markdown source to clipboard.
 */
(function () {
  var DROPDOWN_SELECTOR = ".dropdown-download-buttons";
  var DROPDOWN_MENU_SELECTOR = DROPDOWN_SELECTOR + " .dropdown-menu";
  var COPY_ITEM_ID = "copy-page-source-dropdown-item";
  var COPY_BTN_CLASS = "copy-page-source-btn";
  var COPY_DONE_CLASS = "copy-page-source-done";
  var FEEDBACK_DURATION_MS = 2000;

  function getMarkdownUrl() {
    var dropdown = document.querySelector(DROPDOWN_SELECTOR);
    if (dropdown) {
      var mdLink = dropdown.querySelector('a[href*=".md"]');
      if (mdLink) {
        var href = mdLink.getAttribute("href");
        if (href) {
          try {
            var full = new URL(href, window.location.origin).href;
            if (new URL(full).origin === window.location.origin) return full;
          } catch (e) {
            console.warn("Failed to parse URL:", href, e);
          }
        }
      }
    }
    var pathname = window.location.pathname.replace(/\/$/, "/index.html");
    if (!/\.html?$/.test(pathname)) pathname += ".html";
    var mdPath = pathname.replace(/\.html?$/, ".md").replace(/^\//, "");
    return window.location.origin + "/markdown/" + mdPath;
  }

  function copyMarkdownToClipboard(btn) {
    var url = getMarkdownUrl();
    if (btn) btn.disabled = true;

    fetch(url, { credentials: "same-origin" })
      .then(function (r) {
        if (!r.ok) throw new Error("Markdown not found: " + url);
        return r.text();
      })
      .then(function (text) {
        return navigator.clipboard.writeText(text);
      })
      .then(function () {
        if (btn) {
          var textSpan = btn.querySelector(".btn__text-container");
          if (textSpan) textSpan.textContent = "Copied";
          btn.classList.add(COPY_DONE_CLASS);
          setTimeout(function () {
            if (textSpan) textSpan.textContent = "Copy";
            btn.disabled = false;
            btn.classList.remove(COPY_DONE_CLASS);
          }, FEEDBACK_DURATION_MS);
        }
      })
      .catch(function (err) {
        if (btn) btn.disabled = false;
        console.warn("Copy page source failed:", err);
        alert("Could not copy Markdown. Try opening the .md link and copying manually.");
      });
  }

  function injectCopyOption(menu) {
    if (!menu || document.getElementById(COPY_ITEM_ID)) return;

    var li = document.createElement("li");
    li.id = COPY_ITEM_ID;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-sm btn-download-source-button dropdown-item " + COPY_BTN_CLASS;
    btn.title = "Copy this page's Markdown source to clipboard";
    btn.setAttribute("aria-label", "Copy Markdown to clipboard");
    btn.innerHTML =
      '<span class="btn__icon-container"><i class="fas fa-copy"></i></span>' +
      '<span class="btn__text-container">Copy</span>';
    btn.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      copyMarkdownToClipboard(btn);
    });
    li.appendChild(btn);
    menu.appendChild(li);
  }

  function tryInject() {
    var menu = document.querySelector(DROPDOWN_MENU_SELECTOR);
    if (menu) {
      injectCopyOption(menu);
      return true;
    }
    return false;
  }

  function observeAndInject() {
    if (tryInject()) return;
    var body = document.body;
    if (!body) return;
    var observer = new MutationObserver(function (mutations, obs) {
      if (tryInject()) obs.disconnect();
    });
    observer.observe(body, { childList: true, subtree: true });
  }

  function init() {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", observeAndInject);
    } else {
      observeAndInject();
    }
  }

  init();
})();
