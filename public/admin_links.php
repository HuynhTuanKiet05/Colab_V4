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
                    flash('error', 'Liên kết giữa hai thực thể này đã tồn tại.');
                } else {
                    flash('error', 'Lỗi cơ sở dữ liệu: ' . $e->getMessage());
                }
            }
        } else {
            flash('error', 'Vui lòng chọn đầy đủ thuốc và bệnh.');
        }
    }

    if ($action === 'delete') {
        $id = (int) ($_POST['id'] ?? 0);
        $stmt = db()->prepare('DELETE FROM drug_disease_links WHERE id = :id');
        $stmt->execute(['id' => $id]);
        flash('success', 'Đã xoá liên kết.');
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
    <title>Quản lý liên kết sinh học · AMNTDDA AI</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">Quản lý liên kết sinh học</div>
            <div class="muted">Tạo, kiểm soát và làm sạch các liên kết Thuốc – Bệnh trong cơ sở dữ liệu.</div>
        </div>
        <div class="nav-links">
            <a class="btn btn-ghost" href="admin.php">Quay lại</a>
            <a class="btn btn-danger" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <?php if ($success): ?><div class="alert alert-success"><?= e($success) ?></div><?php endif; ?>
    <?php if ($error): ?><div class="alert alert-error"><?= e($error) ?></div><?php endif; ?>

    <div class="grid split-panel">
        <div class="glass-card">
            <h3>Tạo liên kết Thuốc – Bệnh</h3>
            <p class="muted spacer-lg">Xác lập mối quan hệ để bổ sung dữ liệu huấn luyện, đối chiếu ground truth và phục vụ kiểm chứng.</p>
            <form method="post">
                <input type="hidden" name="action" value="create">
                <div class="stack-tight">
                    <div class="form-group">
                        <label class="label">Chọn thuốc</label>
                        <select class="select" name="drug_id" required>
                            <option value="">-- Chọn thuốc --</option>
                            <?php foreach ($drugs as $d): ?>
                                <option value="<?= $d['id'] ?>"><?= e($d['name']) ?> (<?= e($d['source_code']) ?>)</option>
                            <?php endforeach; ?>
                        </select>
                    </div>

                    <div class="form-group">
                        <label class="label">Chọn bệnh lý</label>
                        <select class="select" name="disease_id" required>
                            <option value="">-- Chọn bệnh --</option>
                            <?php foreach ($diseases as $d): ?>
                                <option value="<?= $d['id'] ?>"><?= e($d['name']) ?> (<?= e($d['source_code']) ?>)</option>
                            <?php endforeach; ?>
                        </select>
                    </div>

                    <div class="form-group">
                        <label class="label">Loại liên kết</label>
                        <select class="select" name="association_type">
                            <option value="known_positive">Đã biết dương tính</option>
                            <option value="known_negative">Đã biết âm tính</option>
                            <option value="predicted">Do mô hình dự đoán</option>
                            <option value="validated">Đã được kiểm chứng</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label class="label">Độ tin cậy</label>
                        <input class="input" type="number" step="0.0001" name="score" value="1.0000">
                    </div>

                    <div class="form-group">
                        <label class="label">Ghi chú</label>
                        <input class="input" name="source_note" placeholder="Ví dụ: PubMed ID, ClinicalTrials.gov...">
                    </div>

                    <button class="btn btn-full" type="submit">Xác lập liên kết</button>
                </div>
            </form>
        </div>

        <div class="glass-card">
            <div class="section-header">
                <div>
                    <h3>Danh sách liên kết thực tế</h3>
                    <p class="muted">Hiển thị các liên kết mới nhất trong cơ sở dữ liệu.</p>
                </div>
                <div class="badge badge-drug"><?= count($links) ?> bản ghi</div>
            </div>

            <div class="table-container table-scroll">
                <table class="table">
                    <thead>
                    <tr>
                        <th>Thuốc</th>
                        <th>Bệnh lý</th>
                        <th>Loại</th>
                        <th>Điểm</th>
                        <th class="align-right">Thao tác</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php if (empty($links)): ?>
                        <tr><td colspan="5" class="center-empty">Chưa có liên kết nào được xác lập.</td></tr>
                    <?php endif; ?>
                    <?php foreach ($links as $row): ?>
                        <tr>
                            <td>
                                <strong><?= e((string) $row['drug_name']) ?></strong><br>
                                <span class="badge badge-drug"><?= e((string) $row['drug_code']) ?></span>
                            </td>
                            <td>
                                <strong><?= e((string) $row['disease_name']) ?></strong><br>
                                <span class="badge badge-disease"><?= e((string) $row['disease_code']) ?></span>
                            </td>
                            <td><span class="badge badge-neutral"><?= e((string) $row['association_type']) ?></span></td>
                            <td class="score-text"><?= e(number_format((float) $row['score'], 4)) ?></td>
                            <td class="align-right">
                                <form method="post" class="inline-form" onsubmit="return confirm('Xoá liên kết này?');">
                                    <input type="hidden" name="action" value="delete">
                                    <input type="hidden" name="id" value="<?= e((string) $row['id']) ?>">
                                    <button class="btn btn-danger btn-sm" type="submit">Xoá</button>
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
