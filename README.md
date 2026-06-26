# ProxyAudit Console (Streamlit)

Bản Streamlit của 2 demo HTML gốc (`auditor_demo.html` và `recommender_demo.html`)
dùng trong paper *"A Comparative Study of Early Fusion and Late Fusion for Fair
AI-Based Resume Screening Systems"*. App gồm 2 tab:

- **Audit dashboard** — bảng so sánh các chiến lược repair, biểu đồ fairness vs
  utility, residual-leakage probe, và ví dụ certificate cho 1 quyết định.
- **Recovery feed & chat** — phát lại (replay) 4 case minh hoạ (`9`, `94`, `0`,
  `20`) kèm khuyến nghị xử lý, và một ô chat "Ask the auditor" để hỏi về từng
  case (có thể dùng Claude API để paraphrase câu trả lời, hoặc dùng fallback
  có sẵn nếu không có API key).

Toàn bộ số liệu (DP gap, AUC, Wilcoxon p, v.v.) được giữ nguyên y như trong 2
file HTML gốc — nằm trong `data.py`.

## Cấu trúc file

```
proxyaudit_streamlit/
├── app.py                          # App Streamlit chính (2 tab)
├── data.py                         # Toàn bộ số liệu đo được, lấy từ HTML gốc
├── requirements.txt
├── .gitignore
└── .streamlit/
    └── secrets.toml.example        # mẫu để bật chat bằng Claude API (tuỳ chọn)
```

## 1. Chạy thử ở máy local

```bash
# (tuỳ chọn) tạo virtual env
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Trình duyệt sẽ tự mở `http://localhost:8501`.

### Bật chat bằng Claude API (tuỳ chọn)

Nếu không làm gì cả, ô "Ask the auditor" vẫn chạy được nhờ fallback có sẵn
(trả lời dựa trên dữ liệu certificate, không cần API key).

Muốn câu trả lời được Claude diễn đạt lại tự nhiên hơn:

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# rồi sửa file .streamlit/secrets.toml, điền API key thật vào
```

`.streamlit/secrets.toml` đã được thêm vào `.gitignore` — **không bao giờ
push file chứa key thật lên GitHub**.

## 2. Đưa code lên GitHub

Giả sử bạn đã có repo trống trên GitHub (ví dụ
`https://github.com/<your-username>/<your-repo>.git`):

```bash
cd proxyaudit_streamlit
git init
git add .
git commit -m "Add ProxyAudit Streamlit console"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Nếu repo đã có sẵn nội dung khác (vd. code FairCV chính), bạn có thể copy
toàn bộ các file ở trên vào một thư mục con trong repo đó (ví dụ
`app_demo/`) rồi commit như bình thường — không cần tạo repo mới.

## 3. Deploy online (để có link demo chạy thật, không cần máy local)

Cách nhanh nhất là dùng **Streamlit Community Cloud** (miễn phí):

1. Vào https://share.streamlit.io → đăng nhập bằng GitHub.
2. Chọn **New app** → chọn repo + branch (`main`) + file chính (`app.py`,
   hoặc `app_demo/app.py` nếu để trong thư mục con).
3. (Tuỳ chọn) Vào **Advanced settings → Secrets**, dán nội dung giống
   `.streamlit/secrets.toml.example` (với key thật) vào đó — không cần file
   secrets trong repo, Streamlit Cloud tự inject lúc deploy.
4. Bấm **Deploy** — sau ~1–2 phút sẽ có link dạng
   `https://<tên-app>.streamlit.app` để chia sẻ hoặc đính kèm vào báo cáo
   capstone / paper.

## File `demo_cases.json` gốc

File JSON gốc bạn gửi đã được hợp nhất vào `data.py` (phần `CASES`), kèm
thêm trường `group` và `eff` (score effect) lấy từ `recommender_demo.html`
để feed minh hoạ vừa có animation vừa có hành động khuyến nghị giống bản
HTML gốc. Nếu sau này bạn muốn nạp case từ một file JSON thật (vd. xuất ra
từ notebook), chỉ cần sửa hàm đọc dữ liệu trong `app.py` để load file đó
thay cho `data.CASES`.
