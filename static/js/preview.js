document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("fileInput");
  const previewContainer = document.getElementById("previewContainer");

  if (!fileInput || !previewContainer) return;

  fileInput.addEventListener("change", () => {
    previewContainer.innerHTML = "";
    const file = fileInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const box = document.createElement("div");
      box.className = "preview-box";

      const img = document.createElement("img");
      img.src = e.target.result;

      const btn = document.createElement("button");
      btn.className = "remove-btn";
      btn.type = "button";
      btn.textContent = "×";
      btn.addEventListener("click", () => {
        fileInput.value = "";
        previewContainer.innerHTML = "";
      });

      box.appendChild(img);
      box.appendChild(btn);
      previewContainer.appendChild(box);
    };
    reader.readAsDataURL(file);
  });
});