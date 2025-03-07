document.addEventListener("DOMContentLoaded", function () {
  initializeThemeToggle();

  initializeFormSubmission();

  initializeTabs();

  initializeShareButton();

  initializeImageGallery();

  initializeTranslation();

  initializeVideoContent();

  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      if (
        tab.dataset.tab === "videos" &&
        typeof twttr !== "undefined" &&
        twttr.widgets
      ) {
        setTimeout(() => {
          twttr.widgets.load();
        }, 100);
      }
    });
  });
});

function initializeThemeToggle() {
  const themeToggle = document.getElementById("themeToggle");
  if (!themeToggle) return;

  const body = document.body;
  const icon = themeToggle.querySelector("i");
  const text = themeToggle.querySelector("span");

  const savedTheme = localStorage.getItem("theme");
  const prefersDark =
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;

  if (savedTheme === "light" || (!savedTheme && !prefersDark)) {
    body.classList.add("light-mode");
    icon.classList.replace("fa-moon", "fa-sun");
    text.textContent = "Light Mode";
  }

  themeToggle.addEventListener("click", () => {
    body.classList.toggle("light-mode");
    const isLight = body.classList.contains("light-mode");

    if (isLight) {
      icon.classList.replace("fa-moon", "fa-sun");
      text.textContent = "Light Mode";
      localStorage.setItem("theme", "light");
    } else {
      icon.classList.replace("fa-sun", "fa-moon");
      text.textContent = "Dark Mode";
      localStorage.setItem("theme", "dark");
    }
  });
}

function initializeFormSubmission() {
  const forms = document.querySelectorAll("form");
  const loadingIndicator = document.getElementById("loadingIndicator");

  forms.forEach((form) => {
    form.addEventListener("submit", function (e) {
      if (form.id === "translateForm") return;

      if (form.id === "compareForm") {
        const url1 = document.getElementById("url1").value.trim();
        const url2 = document.getElementById("url2").value.trim();

        if (!url1 && !url2) {
          e.preventDefault();
          alert("Please enter at least one URL to analyze");
          return;
        }
      }

      if (loadingIndicator) {
        loadingIndicator.style.display = "flex";

        const resultContainer = document.getElementById("resultContainer");
        if (resultContainer) resultContainer.style.display = "none";
      }
    });
  });
}

function initializeTabs() {
  const tabs = document.querySelectorAll(".tab");
  if (!tabs.length) return;

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const targetId = `${tab.dataset.tab}-tab`;
      const targetContent = document.getElementById(targetId);

      document
        .querySelectorAll(".tab")
        .forEach((t) => t.classList.remove("active"));
      document
        .querySelectorAll(".tab-content")
        .forEach((c) => c.classList.remove("active"));

      tab.classList.add("active");
      if (targetContent) targetContent.classList.add("active");
    });
  });
}

function initializeShareButton() {
  const shareButton = document.getElementById("shareButton");
  if (!shareButton) return;

  shareButton.addEventListener("click", async () => {
    try {
      if (navigator.share) {
        await navigator.share({
          title:
            document.querySelector(".article-title")?.textContent ||
            "Article Analysis",
          text: "Check out this article analysis!",
          url: window.location.href,
        });
      } else {
        await navigator.clipboard.writeText(window.location.href);
        alert("Link copied to clipboard!");
      }
    } catch (err) {
      console.error("Error sharing:", err);
    }
  });
}

function initializeImageGallery() {
  const galleryImages = document.querySelectorAll(".gallery-image");
  if (!galleryImages.length) return;

  const modal = document.createElement("div");
  modal.className = "image-modal";
  modal.style.display = "none";

  const modalContent = document.createElement("img");
  modalContent.className = "modal-content";

  const closeButton = document.createElement("span");
  closeButton.className = "close-modal";
  closeButton.innerHTML = "&times;";

  modal.appendChild(closeButton);
  modal.appendChild(modalContent);
  document.body.appendChild(modal);

  galleryImages.forEach((img) => {
    img.addEventListener("click", () => {
      modalContent.src = img.src;
      modal.style.display = "flex";
    });
  });

  closeButton.addEventListener("click", () => {
    modal.style.display = "none";
  });

  window.addEventListener("click", (e) => {
    if (e.target === modal) {
      modal.style.display = "none";
    }
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.style.display !== "none") {
      modal.style.display = "none";
    }
  });
}

function initializeTranslation() {
  const translationSelectors = document.querySelectorAll('[id$="-language"]');
  if (!translationSelectors.length) return;

  translationSelectors.forEach((selector) => {
    const contentType = selector.id.split("-")[0];
    const contentElement = document.getElementById(`${contentType}-content`);

    if (!contentElement) return;

    contentElement.dataset.originalText = contentElement.textContent;

    // Set initial selection to match the detected language if available
    if (contentElement.dataset.sourceLang) {
      // If we have options for the source language, select it
      const sourceOption = selector.querySelector(
        `option[value="${contentElement.dataset.sourceLang}"]`
      );
      if (sourceOption) {
        sourceOption.selected = true;
      }
    }

    selector.addEventListener("change", () => {
      translateContent(contentType, selector.value);
    });
  });
}

function translateContent(contentType, targetLang) {
  const contentElement = document.getElementById(`${contentType}-content`);
  if (!contentElement) return;

  const originalText =
    contentElement.dataset.originalText || contentElement.textContent;
  const sourceLang = contentElement.dataset.sourceLang || null;
  const selector = document.getElementById(`${contentType}-language`);

  if (!targetLang && selector) {
    targetLang = selector.value;
  }

  // If target is the source language, restore original text
  if (targetLang === sourceLang) {
    contentElement.textContent = originalText;
    return;
  }

  const originalContent = contentElement.innerHTML;
  contentElement.innerHTML =
    '<div class="loading-spinner" style="margin: 10px auto;"></div>';

  fetch("/translate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text: originalText,
      target: targetLang,
      source: sourceLang,
    }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Translation request failed");
      }
      return response.json();
    })
    .then((data) => {
      if (data.translated) {
        contentElement.textContent = data.translated;

        // If source was detected, update the data attribute
        if (data.source_language) {
          contentElement.dataset.sourceLang = data.source_language;
        }
      } else {
        throw new Error("No translation received");
      }
    })
    .catch((error) => {
      console.error("Translation error:", error);
      contentElement.innerHTML = originalContent;
      alert("Translation failed. Please try again later.");
    });
}

function initializeVideoContent() {
  if (document.querySelector(".twitter-embed")) {
    if (typeof twttr === "undefined") {
      const script = document.createElement("script");
      script.src = "https://platform.twitter.com/widgets.js";
      script.async = true;
      document.head.appendChild(script);
    } else if (typeof twttr !== "undefined" && twttr.widgets) {
      twttr.widgets.load();
    }
  }

  const videos = document.querySelectorAll(
    ".video-container iframe, .video-container video"
  );
  videos.forEach((video) => {
    video.setAttribute("loading", "lazy");

    if (video.tagName === "VIDEO") {
      video.setAttribute("preload", "none");
      video.setAttribute("poster", "/static/video-placeholder.png");
    }
  });
}
