/**
 * Add "Copy" option inside the download dropdown (.md, .pdf, Copy).
 * Copies the current page's Markdown source to clipboard.
 */
(function () {
  function getMarkdownUrl() {
    var dropdown = document.querySelector(".dropdown-download-buttons");
    if (dropdown) {
      var mdLink = dropdown.querySelector('a[href*=".md"]');
      if (mdLink) {
        var href = mdLink.getAttribute("href");
        if (href) {
          try {
            var full = new URL(href, window.location.origin).href;
            if (new URL(full).origin === window.location.origin) return full;
          } catch (e) {}
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
          btn.classList.add("copy-page-source-done");
          setTimeout(function () {
            if (textSpan) textSpan.textContent = "Copy";
            btn.disabled = false;
            btn.classList.remove("copy-page-source-done");
          }, 2000);
        }
      })
      .catch(function (err) {
        if (btn) btn.disabled = false;
        console.warn("Copy page source failed:", err);
        alert("Could not copy Markdown. Try opening the .md link and copying manually.");
      });
  }

  function injectCopyOption() {
    var menu = document.querySelector(".dropdown-download-buttons .dropdown-menu");
    if (!menu || document.getElementById("copy-page-source-dropdown-item")) return;

    var li = document.createElement("li");
    li.id = "copy-page-source-dropdown-item";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-sm btn-download-source-button dropdown-item copy-page-source-btn";
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

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", injectCopyOption);
  } else {
    injectCopyOption();
  }
  setTimeout(injectCopyOption, 500);
  setTimeout(injectCopyOption, 1500);
})();
