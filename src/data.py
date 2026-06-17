# -*- coding: utf-8 -*-
"""Nạp dữ liệu + metadata (tên cột tiếng Việt, nhóm đặc trưng, bảng giải mã)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TARGET   = "86"
FEATURES = [str(i) for i in range(1, 86)]
SOCIO    = [str(i) for i in range(1, 43)]
PP       = ["43"]
CONTRIB  = [str(i) for i in range(44, 65)]   # Đóng phí 21 loại BH
NUMBER   = [str(i) for i in range(65, 86)]   # Số HĐ 21 loại BH
HICARD   = ["1", "5"]                          # subtype (41 mức) + main type (10 mức)

SOCIO_NAMES = {
 1:"Phân nhóm KH (subtype)", 2:"Số nhà", 3:"Quy mô hộ TB", 4:"Tuổi TB",
 5:"Nhóm KH chính (main type)", 6:"Công giáo La Mã", 7:"Tin Lành",
 8:"Tôn giáo khác", 9:"Không tôn giáo", 10:"Đã kết hôn", 11:"Sống chung",
 12:"Quan hệ khác", 13:"Độc thân", 14:"Hộ không con", 15:"Hộ có con",
 16:"Học vấn cao", 17:"Học vấn TB", 18:"Học vấn thấp", 19:"Địa vị cao",
 20:"Doanh nhân", 21:"Nông dân", 22:"Quản lý cấp trung", 23:"LĐ có tay nghề",
 24:"LĐ phổ thông", 25:"Tầng lớp A", 26:"Tầng lớp B1", 27:"Tầng lớp B2",
 28:"Tầng lớp C", 29:"Tầng lớp D", 30:"Nhà thuê", 31:"Sở hữu nhà",
 32:"Có 1 ô tô", 33:"Có 2 ô tô", 34:"Không ô tô", 35:"BHYT quốc gia",
 36:"BHYT tư nhân", 37:"Thu nhập <30k", 38:"Thu nhập 30-45k",
 39:"Thu nhập 45-75k", 40:"Thu nhập 75-122k", 41:"Thu nhập >123k",
 42:"Thu nhập TB", 43:"Hạng sức mua",
}
INS_TYPES = [
 "TN bên thứ ba (cá nhân)", "TN bên thứ ba (DN)", "TN bên thứ ba (nông nghiệp)",
 "ô tô", "xe tải nhẹ", "mô tô/scooter", "xe tải lớn", "rơ-moóc", "máy kéo",
 "máy nông nghiệp", "xe máy điện (moped)", "nhân thọ", "tai nạn cá nhân",
 "tai nạn gia đình", "khuyết tật", "cháy nổ", "ván lướt sóng", "thuyền",
 "xe đạp", "tài sản", "an sinh xã hội",
]
NAME = {str(c): nm for c, nm in SOCIO_NAMES.items()}
for i, t in enumerate(INS_TYPES):
    NAME[str(44 + i)] = "Đóng phí BH " + t
    NAME[str(65 + i)] = "Số HĐ BH " + t

A2 = {1:"20-30", 2:"30-40", 3:"40-50", 4:"50-60", 5:"60-70", 6:"70-80"}
A3 = {1:"Hedonist thành đạt", 2:"Người vươn lên", 3:"Gia đình trung bình", 4:"Độc lập sự nghiệp",
      5:"Sống sung túc", 6:"Cao tuổi an nhàn", 7:"Nghỉ hưu & sùng đạo", 8:"Gia đình con đã lớn",
      9:"Gia đình bảo thủ", 10:"Nông dân"}


def disp(col):
    """Tên hiển thị tiếng Việt cho 1 cột (kể cả đặc trưng kỹ thuật agg_*/ix_*)."""
    return NAME.get(str(col), str(col))


def load_data(root: Path = ROOT):
    """Trả về (train, test). Cột giữ nguyên tên gốc 'ID','1'..'86'."""
    train = pd.read_csv(Path(root) / "data" / "train_data.txt")
    test  = pd.read_csv(Path(root) / "data" / "test_data.txt")
    return train, test
