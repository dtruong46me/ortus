# Sổ tay CS336

Nguồn của artifact: https://claude.ai/artifact/LsuUUkFNZJXtL7iEha7qnA

Thêm một lecture mới:
1. Xuất ảnh slide: `pdftoppm -jpeg -jpegopt quality=72 -scale-to-x 1400 -scale-to-y -1 lecture_XX.pdf img/lXX/s`
2. Viết `lectures/lXX.html` theo mẫu `lectures/l04.html` hoặc `lectures/l05.html` (`<h2 id="lXX-...">`, `<div class="slides" data-s="2,3">`, các khung `.note.key/.life/.warn/.num`, `.quiz`).
3. Thêm một mục vào `lectures.json`.
4. Đăng lại `index.html` lên cùng URL artifact, kèm các file mới.

Giao diện (`index.html`):
- Thanh trên: chọn lecture, mở/đóng bảng Notes, đổi Light/Auto/Dark.
- Trái: mục lục phân cấp (Phần → mục h2 → mục h3) của lecture đang mở.
- Phải: Notes. Bôi đen một đoạn trong bài để highlight hoặc thêm ghi chú. Ghi chú neo theo đoạn trích + vài chữ trước/sau.
- Ghi chú lưu trong kho dữ liệu riêng của từng người (artifact capability `db` + `user`, đường dẫn `data/users/<id>/<noteId>`). Nếu không có kho đó thì lưu ở `localStorage`. Khi đăng lại, cứ bỏ trống `capabilities` để giữ nguyên khai báo cũ.
