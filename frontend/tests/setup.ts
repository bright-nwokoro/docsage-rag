import "@testing-library/jest-dom/vitest";

// Polyfill scrollTo for jsdom
Element.prototype.scrollTo = function () {
  // no-op
};

