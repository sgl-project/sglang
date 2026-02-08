var labels_by_text = {};

function ready() {
  var li = document.getElementsByClassName("tab-label");
  for (const label of li) {
    label.onclick = onLabelClick;
    const text = label.textContent;
    if (!labels_by_text[text]) {
      labels_by_text[text] = [];
    }
    labels_by_text[text].push(label);
  }
}

function onLabelClick() {
  // Activate other labels with the same text.
  for (label of labels_by_text[this.textContent]) {
    label.previousSibling.checked = true;
  }
}
document.addEventListener("DOMContentLoaded", ready, false);
