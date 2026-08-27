(function () {
    "use strict";

    var oldOrigin = "https://sgl-project.github.io";
    var newOrigin = "https://docs.sglang.io";

    function buildNewDocsUrl() {
        var href = window.location.href;

        if (href === oldOrigin || href.indexOf(oldOrigin + "/") === 0) {
            return href.replace(oldOrigin, newOrigin);
        }

        return newOrigin + window.location.pathname + window.location.search + window.location.hash;
    }

    function addDeprecationBanner() {
        if (document.getElementById("sglang-docs-deprecation-banner")) {
            return;
        }

        var link = document.createElement("a");
        link.href = buildNewDocsUrl();
        link.textContent = link.href;

        var banner = document.createElement("div");
        banner.id = "sglang-docs-deprecation-banner";
        banner.className = "sglang-docs-deprecation-banner";
        banner.setAttribute("role", "status");
        banner.setAttribute("aria-live", "polite");

        var prefix = document.createTextNode(
            "This legacy documentation site will be deprecated soon. Please use the new SGLang documentation at "
        );
        var suffix = document.createTextNode(".");

        banner.appendChild(prefix);
        banner.appendChild(link);
        banner.appendChild(suffix);

        document.body.insertBefore(banner, document.body.firstChild);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", addDeprecationBanner);
    } else {
        addDeprecationBanner();
    }
})();
