// FlowGuard static site interactions.
// Keep this file lightweight for GitHub Pages.

const nav = document.querySelector(".nav");

window.addEventListener("scroll", () => {
  if (window.scrollY > 24) {
    nav?.classList.add("is-scrolled");
  } else {
    nav?.classList.remove("is-scrolled");
  }
});
