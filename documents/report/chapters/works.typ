#import "@preview/lovelace:0.3.0": *

= Truy vấn ảnh theo đặc trưng

Hệ thống cho phép truy vấn ảnh từ tập dữ liệu TMBuD bằng nhiều loại đặc trưng
khác nhau. Với mỗi ảnh truy vấn, hệ thống trích xuất đặc trưng tương ứng, so
khớp với toàn bộ cơ sở dữ liệu và hiển thị các ảnh có độ tương đồng cao nhất theo
thứ tự giảm dần, kèm theo khoảng cách so khớp và thời gian truy vấn. Quy trình
truy vấn cơ bản (quét tuyến tính) được mô tả như sau:

#figure(
  pseudocode-list[
    + *Input:* ảnh truy vấn $q$, CSDL đặc trưng $D$, số kết quả $k$
    + $f_q <-$ trích xuất đặc trưng của $q$
    + *for* mỗi ảnh $d in D$ *do*
      + $d_"dist" <- "distance"(f_q, f_d)$
    + *end*
    + sắp xếp $D$ theo $d_"dist"$ tăng dần
    + *return* $k$ ảnh có khoảng cách nhỏ nhất
  ],
  caption: [Thuật toán truy vấn brute-force],
)

Phần dưới đây minh hoạ kết quả truy vấn của từng đặc trưng trên cùng một giao
diện chương trình.

// Helper: hiển thị ba ảnh kết quả truy vấn cạnh nhau.
#let demo3(a, b, c, cap) = figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 6pt,
    image(a, width: 100%), image(b, width: 100%), image(c, width: 100%),
  ),
  caption: cap,
)

== Color Histogram

#demo3(
  "../assets/images/features/color_hist_1.png",
  "../assets/images/features/color_hist_2.png",
  "../assets/images/features/color_hist_3.png",
  [Kết quả truy vấn bằng Color Histogram],
)

== Color Correlogram

#demo3(
  "../assets/images/features/correlogram_1.png",
  "../assets/images/features/correlogram_2.png",
  "../assets/images/features/correlogram_3.png",
  [Kết quả truy vấn bằng Color Correlogram],
)

== SIFT

#demo3(
  "../assets/images/features/sift_1.png",
  "../assets/images/features/sift_2.png",
  "../assets/images/features/sift_3.png",
  [Kết quả truy vấn bằng SIFT],
)

== ORB

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 6pt,
    image("../assets/images/features/orb_1.png", width: 100%),
    image("../assets/images/features/orb_2.png", width: 100%),
  ),
  caption: [Kết quả truy vấn bằng ORB],
)

== Hu Moments

#demo3(
  "../assets/images/features/hu_1.png",
  "../assets/images/features/hu_2.png",
  "../assets/images/features/hu_3.png",
  [Kết quả truy vấn bằng Hu Moments],
)

= Cải tiến hệ thống

Ngoài các đặc trưng cơ bản, hệ thống được bổ sung một số cải tiến nhằm nâng cao
độ chính xác và tốc độ truy vấn.

== Nâng cao độ chính xác

- *Tiền xử lý ảnh:* trước khi trích xuất đặc trưng, ảnh được đưa về cùng kích
  thước và cân bằng độ tương phản bằng CLAHE trên kênh sáng (không gian LAB),
  giúp giảm ảnh hưởng của điều kiện ánh sáng khác nhau (ngày/đêm) trong tập dữ
  liệu.

- *Bổ sung đặc trưng hình dạng, biên và vân:* thêm Hu Moments (mô tả hình dạng
  tổng thể), LBP (mô tả vân/texture) và HOG (mô tả biên và hướng gradient) bên
  cạnh các đặc trưng màu và cục bộ.

- *Kết hợp có trọng số ở mức khoảng cách (combinedw):* thay vì ghép các đặc
  trưng thành một vector (cách này buộc dùng chung một độ đo và bị các đặc trưng
  yếu làm nhiễu), hệ thống chỉ chọn ba đặc trưng mạnh nhất (Color Histogram,
  Color Correlogram, LBP). Với mỗi đặc trưng, khoảng cách từ ảnh truy vấn tới
  toàn bộ CSDL được tính bằng _đúng độ đo phù hợp_ của nó (Chi-square cho biểu
  đồ, L2 cho phần còn lại), rồi chuẩn hoá về cùng thang $[0, 1]$ và cộng lại theo
  trọng số bằng MAP\@3 riêng lẻ của từng đặc trưng. Nhờ giữ đúng độ đo và loại bỏ
  các đặc trưng yếu, cách này cho độ chính xác cao hơn hẳn mọi đặc trưng đơn lẻ
  (xem phần Đánh giá kết quả).

#figure(
  pseudocode-list[
    + *Input:* ảnh truy vấn $q$, CSDL $D$, tập đặc trưng mạnh
      $F = {f_1, ..., f_m}$ với trọng số $w_1, ..., w_m$
    + $s[j] <- 0$ với mọi ảnh $d_j in D$
    + *for* mỗi đặc trưng $f_i in F$ *do*
      + *for* mỗi ảnh $d_j in D$ *do*
        + $t_i [j] <- "distance"_(f_i) (q, d_j)$ #h(0.5em) (độ đo riêng của $f_i$)
      + *end*
      + chuẩn hoá $t_i$ về thang $[0, 1]$ (min–max)
      + $s[j] <- s[j] + w_i dot t_i [j]$ với mọi $j$
    + *end*
    + sắp xếp $D$ theo $s$ tăng dần và *return* $k$ ảnh đầu
  ],
  caption: [Kết hợp có trọng số ở mức khoảng cách (combinedw)],
)

== Tăng tốc độ truy vấn

- *Chỉ mục FLANN dựng sẵn:* thay vì quét tuyến tính toàn bộ cơ sở dữ liệu với độ
  phức tạp $O(N)$, hệ thống xây dựng sẵn chỉ mục cây kd (kd-tree) của FLANN và
  tìm kiếm gần đúng với độ phức tạp khoảng $O(log N)$, rút ngắn đáng kể thời gian
  truy vấn trên tập dữ liệu lớn.

- *Hiển thị thời gian truy vấn:* mỗi lần truy vấn đều đo và hiển thị thời gian
  thực hiện trên giao diện, thuận tiện cho việc so sánh giữa cách quét tuyến tính
  và cách dùng chỉ mục FLANN.

#figure(
  pseudocode-list[
    + *Dựng chỉ mục (một lần):* $T <- "buildKdTree"(D)$
    + *Truy vấn:*
    + $f_q <-$ trích xuất đặc trưng của $q$
    + $R <- "knnSearch"(T, f_q, k)$ #h(1em) (tìm gần đúng, $O(log N)$)
    + *return* $R$
  ],
  caption: [Truy vấn dùng chỉ mục FLANN dựng sẵn],
)

= Đánh giá kết quả

Hệ thống được đánh giá bằng độ đo MAP (Mean Average Precision) trên tập dữ liệu
TMBuD. Hai ảnh được xem là liên quan nếu cùng thuộc một toà nhà (theo cột
_Building Name_ trong tập dữ liệu). Việc đánh giá thực hiện trên 1358 ảnh truy
vấn thuộc 133 toà nhà, với số lượng kết quả trả về $k = 3, 5, 11, 21$. Toàn bộ
quy trình đánh giá được tự động hoá bằng chương trình `benchmark.py` để bảo đảm
khả năng tái lập.

== Độ chính xác của các đặc trưng

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (left, center, center, center, center),
    table.header([*Đặc trưng*], [*MAP\@3*], [*MAP\@5*], [*MAP\@11*], [*MAP\@21*]),
    table.cell(fill: rgb("#FFF1A8"))[*Color Histogram*],
    table.cell(fill: rgb("#FFF1A8"))[*0.3808*],
    table.cell(fill: rgb("#FFF1A8"))[*0.2843*],
    table.cell(fill: rgb("#FFF1A8"))[*0.2119*],
    table.cell(fill: rgb("#FFF1A8"))[*0.2002*],
    [Color Correlogram], [0.3101], [0.2320], [0.1686], [0.1598],
    [SIFT], [0.2521], [0.1785], [0.1310], [0.1267],
    [ORB], [0.1727], [0.1260], [0.0927], [0.0892],
  ),
  caption: [MAP của các đặc trưng cơ bản (không tiền xử lý); hàng tô vàng là
    kết quả tốt nhất],
)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (left, center, center, center, center),
    table.header([*Đặc trưng*], [*MAP\@3*], [*MAP\@5*], [*MAP\@11*], [*MAP\@21*]),
    [Color Histogram], [0.3754], [0.2810], [0.2083], [0.1972],
    [Color Correlogram], [0.3190], [0.2383], [0.1754], [0.1645],
    [SIFT], [0.2211], [0.1623], [0.1180], [0.1143],
    [ORB], [0.1033], [0.0748], [0.0565], [0.0569],
    [Hu Moments], [0.0293], [0.0215], [0.0160], [0.0157],
    [LBP], [0.2819], [0.2076], [0.1517], [0.1461],
    [HOG], [0.2253], [0.1558], [0.1047], [0.0985],
    [Kết hợp ghép vector (combined)], [0.2135], [0.1478], [0.0996], [0.0898],
    table.cell(fill: rgb("#FFF1A8"))[*Kết hợp có trọng số (combinedw)*],
    table.cell(fill: rgb("#FFF1A8"))[*0.4326*],
    table.cell(fill: rgb("#FFF1A8"))[*0.3310*],
    table.cell(fill: rgb("#FFF1A8"))[*0.2454*],
    table.cell(fill: rgb("#FFF1A8"))[*0.2312*],
  ),
  caption: [MAP trên cơ sở dữ liệu đã tiền xử lý; hàng tô vàng (combinedw) là
    kết quả tốt nhất],
)

#figure(
  image("../assets/images/map_plot.png", width: 100%),
  caption: [MAP của từng đặc trưng; kết hợp có trọng số (combinedw) cho kết quả
    tốt nhất (được đánh dấu)],
)

*Nhận xét:* biểu đồ màu là đặc trưng đơn lẻ tốt nhất, còn cách ghép vector
(combined) lại _làm giảm_ độ chính xác do các đặc trưng yếu gây nhiễu và buộc
dùng chung một khoảng cách. Kết hợp có trọng số (combinedw) khắc phục điều này và
đạt MAP\@3 = 0.4326, *vượt mọi đặc trưng đơn lẻ* (cao hơn Color Histogram khoảng
$+15%$) ở mọi mức $k$.

== Tốc độ truy vấn

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 6pt,
    align: (left, center, center),
    table.header([*Phương pháp*], [*Thời gian trung vị (ms)*], [*Tăng tốc*]),
    [Quét tuyến tính (brute-force)], [2.013], [1.0#sym.times],
    table.cell(fill: rgb("#FFF1A8"))[*Chỉ mục FLANN (kd-tree)*],
    table.cell(fill: rgb("#FFF1A8"))[*0.140*],
    table.cell(fill: rgb("#FFF1A8"))[*14.4#sym.times*],
  ),
  caption: [So sánh thời gian truy vấn với đặc trưng combined; hàng tô vàng là
    phương pháp nhanh nhất],
)

#figure(
  image("../assets/images/speed_plot.png", width: 62%),
  caption: [Chỉ mục FLANN nhanh hơn khoảng 14 lần so với quét tuyến tính],
)

*Nhận xét:* FLANN cho cùng thứ hạng kết quả nhưng nhanh hơn khoảng 14 lần.
