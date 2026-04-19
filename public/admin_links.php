<?php
require_once __DIR__ . '/../app/bootstrap.php';
require_admin();

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? 'create';

    if ($action === 'create') {
        $drugId = (int) ($_POST['drug_id'] ?? 0);
        $diseaseId = (int) ($_POST['disease_id'] ?? 0);
        $assocType = $_POST['association_type'] ?? 'known_positive';
        $score = (float) ($_POST['score'] ?? 1.0);
        $note = trim($_POST['source_note'] ?? '');

        if ($drugId > 0 && $diseaseId > 0) {
            try {
                $stmt = db()->prepare('INSERT INTO drug_disease_links (drug_id, disease_id, association_type, score, source_note) VALUES (:drug_id, :disease_id, :association_type, :score, :source_note)');
                $stmt->execute([
                    'drug_id' => $drugId,
                    'disease_id' => $diseaseId,
                    'association_type' => $assocType,
                    'score' => $score,
                    'source_note' => $note
                ]);
                flash('success', 'Đã tạo liên kết mới thành công.');
            } catch (PDOException $e) {
                if ($e->getCode() == 23000) {
                    flash('error', 'Lỗi: Liên kết giữa hai thực thể này đã tồn tại.');
                } else {
                    flash('error', 'Lỗi CSDL: ' . $e->getMessage());
                }
            }
        } else {
            flash('error', 'Vui lòng chọn đầy đủ Thuốc và Bệnh.');
        }
    }

    if ($action === 'delete') {
        $id = (int) ($_POST['id'] ?? 0);
        $stmt = db()->prepare('DELETE FROM drug_disease_links WHERE id = :id');
        $stmt->execute(['id' => $id]);
        flash('success', 'Đã xóa liên kết.');
    }

    redirect('admin_links.php');
}

$success = flash('success');
$error = flash('error');

$drugs = db()->query('SELECT id, name, source_code FROM drugs ORDER BY name ASC')->fetchAll();
$diseases = db()->query('SELECT id, name, source_code FROM diseases ORDER BY name ASC')->fetchAll();

$links = db()->query('
    SELECT l.*, dr.name as drug_name, dr.source_code as drug_code, di.name as disease_name, di.source_code as disease_code 
    FROM drug_disease_links l
    JOIN drugs dr ON l.drug_id = dr.id
    JOIN diseases di ON l.disease_id = di.id
    ORDER BY l.created_at DESC LIMIT 50
')->fetchAll();
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin - Quản lý liên kết sinh học</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">Quản lý liên kết sinh học</div>
            <div class="muted">Tạo ground truth cho mô hình và kiểm soát dữ liệu quan hệ</div>
        </div>
        <div class="nav-links">
            <a class="btn" style="background: rgba(255,255,255,0.06); box-shadow:none;" href="admin.php">Quay lại Quản trị</a>
            <a class="btn" style="background: rgba(244, 63, 94, 0.16); box-shadow:none;" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <?php if ($success): ?><div class="alert alert-success"><?= e($success) ?></div><?php endif; ?>
    <?php if ($error): ?><div class="alert alert-error"><?= e($error) ?></div><?php endif; ?>

    <div class="grid grid-2" style="grid-template-columns: 430px 1fr;">
        <div class="glass-card">
            <h3>Tạo liên kết Thuốc - Bệnh</h3>
            <p class="muted" style="margin-bottom:20px;">Xác lập mối quan hệ để kiểm chứng hoặc bổ sung dữ liệu huấn luyện.</p>
            <form method="post">
                <input type="hidden" name="action" value="create">
                <div style="display:grid; gap:14px;">
                    <div class="form-group">
                        <label class="label">Chọn Thuốc</label>
                        <select class="select" name="drug_id" required>
                            <option value="">-- Chọn thuốc --</option>
                            <?php foreach ($drugs as $d): ?><option value="<?= $d['id'] ?>"><?= e($d['name']) ?> (<?= e($d['source_code']) ?>)</option><?php endforeach; ?>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="label">Chọn Bệnh lý</label>
                        <select class="select" name="disease_id" required>
                            <option value="">-- Chọn bệnh --</option>
                            <?php foreach ($diseases as $d): ?><option value="<?= $d['id'] ?>"><?= e($d['name']) ?> (<?= e($d['source_code']) ?>)</option><?php endforeach; ?>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="label">Loại liên kết</label>
                        <select class="select" name="association_type">
                            <option value="known_positive">Đã biết (Dương tính)</option>
                            <option value="known_negative">Đã biết (Âm tính)</option>
                            <option value="predicted">Do mô hình dự đoán</option>
                            <option value="validated">Đã được kiểm chứng lab</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="label">Độ tin cậy (Score 0-1)</label>
                        <input class="input" type="number" step="0.0001" name="score" value="1.0000">
                    </div>
                    <div class="form-group">
                        <label class="label">Ghi chú nguồn dữ liệu</label>
                        <input class="input" name="source_note" placeholder="Ví dụ: PubMed ID, ClinicalTrials.gov...">
                    </div>
                    <button class="btn" type="submit" style="width:100%;">Xác lập liên kết</button>
                </div>
            </form>
        </div>

        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:end; margin-bottom:14px; gap:12px; flex-wrap:wrap;">
                <div>
                    <h3>Danh sách liên kết thực tế</h3>
                    <p class="muted">Hiển thị các liên kết gần nhất trong cơ sở dữ liệu.</p>
                </div>
                <div class="badge badge-drug"><?= count($links) ?> records</div>
            </div>
            <div class="table-container" style="max-height: 700px; overflow-y:auto;">
                <table class="table">
                    <thead>
                        <tr><th>Thuốc</th><th>Bệnh lý</th><th>Loại</th><th>Score</th><th style="text-align:right;">Hành động</th></tr>
                    </thead>
                    <tbody>
                    <?php if (empty($links)): ?><tr><td colspan="5" style="text-align:center; padding:40px;" class="muted">Chưa có liên kết nào được xác lập</td></tr><?php endif; ?>
                    <?php foreach ($links as $row): ?>
                        <tr>
                            <td><strong><?= e((string)$row['drug_name']) ?></strong><br><span class="badge badge-drug" style="font-size: 9px;"><?= e((string)$row['drug_code']) ?></span></td>
                            <td><strong><?= e((string)$row['disease_name']) ?></strong><br><span class="badge badge-disease" style="font-size: 9px;"><?= e((string)$row['disease_code']) ?></span></td>
                            <td><span class="badge" style="background: rgba(255,255,255,0.05); color:#cbd5e1; border:1px solid rgba(148,163,184,0.16);"><?= e((string)$row['association_type']) ?></span></td>
                            <td class="score-text"><?= e(number_format((float)$row['score'], 4)) ?></td>
                            <td style="text-align:right;">
                                <form method="post" onsubmit="return confirm('Xóa liên kết này?');" style="display:inline;">
                                    <input type="hidden" name="action" value="delete">
                                    <input type="hidden" name="id" value="<?= e((string)$row['id']) ?>">
                                    <button class="btn" type="submit" style="height:40px; padding:0 14px; background: linear-gradient(135deg, #ef4444, #f97316);">Xóa</button>
                                </form>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</div>
</body>
</html>