# Sổ tay CS336

Nguồn của artifact: https://claude.ai/artifact/LsuUUkFNZJXtL7iEha7qnA

Thêm một lecture mới:
1. Xuất ảnh slide: `pdftoppm -jpeg -jpegopt quality=72 -scale-to-x 1400 -scale-to-y -1 lecture_XX.pdf img/lXX/s`
2. Viết `lectures/lXX.html` theo mẫu `lectures/l04.html` (`<h2 id="lXX-...">`, `<div class="slides" data-s="2,3">`, các khung `.note.key/.life/.warn/.num`, `.quiz`).
3. Thêm một mục vào `lectures.json`.
4. Đăng lại `index.html` lên cùng URL artifact, kèm các file mới.
