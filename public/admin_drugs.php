<?php
require_once __DIR__ . '/../app/bootstrap.php';
require_admin();

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? 'create';
    $sourceCode = trim($_POST['source_code'] ?? '');
    $name = trim($_POST['name'] ?? '');
    $smiles = trim($_POST['smiles'] ?? '');
    $description = trim($_POST['description'] ?? '');

    if ($action === 'create' && $sourceCode !== '' && $name !== '') {
        $stmt = db()->prepare('INSERT INTO drugs (source_code, name, smiles, description) VALUES (:source_code, :name, :smiles, :description)');
        $stmt->execute([
            'source_code' => $sourceCode,
            'name' => $name,
            'smiles' => $smiles,
            'description' => $description
        ]);
        flash('success', 'Đã thêm thuốc mới.');
    }

    if ($action === 'delete') {
        $id = (int) ($_POST['id'] ?? 0);
        $stmt = db()->prepare('DELETE FROM drugs WHERE id = :id');
        $stmt->execute(['id' => $id]);
        flash('success', 'Đã xóa thuốc.');
    }

    redirect('admin_drugs.php');
}

$success = flash('success');
$rows = db()->query('SELECT * FROM drugs ORDER BY created_at DESC LIMIT 50')->fetchAll();
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin - Quản lý thuốc</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">Quản lý thuốc</div>
            <div class="muted">Thêm, xoá và theo dõi thực thể thuốc</div>
        </div>
        <div class="nav-links">
            <a class="btn" style="background: rgba(255,255,255,0.06); box-shadow:none;" href="admin.php">Quay lại Quản trị</a>
            <a class="btn" style="background: rgba(244, 63, 94, 0.16); box-shadow:none;" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <?php if ($success): ?><div class="alert alert-success"><?= e($success) ?></div><?php endif; ?>

    <div class="grid grid-2" style="grid-template-columns: 430px 1fr;">
        <div class="glass-card">
            <h3>Thêm thực thể Thuốc mới</h3>
            <p class="muted" style="margin-bottom: 18px;">Điền thông tin định danh và đặc trưng hóa học.</p>
            <form method="post">
                <input type="hidden" name="action" value="create">
                <div style="display:grid; gap:14px;">
                    <div class="form-group"><label class="label">Mã nguồn (Source Code)</label><input class="input" name="source_code" placeholder="Ví dụ: DB00014" required></div>
                    <div class="form-group"><label class="label">Tên thuốc</label><input class="input" name="name" placeholder="Ví dụ: Goserelin" required></div>
                    <div class="form-group"><label class="label">Cấu trúc SMILES</label><textarea class="input" name="smiles" placeholder="Dành cho tính toán đặc trưng phân tử..."></textarea></div>
                    <div class="form-group"><label class="label">Mô tả chi tiết</label><textarea class="input" name="description" placeholder="Thông tin bổ sung..."></textarea></div>
                    <button class="btn" type="submit" style="width:100%;">Lưu thông tin thực thể</button>
                </div>
            </form>
        </div>

        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:end; gap:12px; margin-bottom: 14px; flex-wrap: wrap;">
                <div>
                    <h3>Danh sách thuốc trong CSDL</h3>
                    <p class="muted">Hiển thị tối đa 50 bản ghi gần nhất.</p>
                </div>
                <div class="badge badge-drug"><?= count($rows) ?> records</div>
            </div>
            <div class="table-container" style="max-height: 680px; overflow-y: auto;">
                <table class="table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Mã nguồn</th>
                            <th>Tên thuốc</th>
                            <th style="text-align:right;">Hành động</th>
                        </tr>
                    </thead>
                    <tbody>
                    <?php if (empty($rows)): ?>
                        <tr><td colspan="4" style="text-align:center; padding: 40px;" class="muted">Chưa có dữ liệu thuốc</td></tr>
                    <?php endif; ?>
                    <?php foreach ($rows as $row): ?>
                        <tr>
                            <td class="muted">#<?= $row['id'] ?></td>
                            <td><span class="badge badge-drug"><?= e((string) $row['source_code']) ?></span></td>
                            <td style="font-weight:600;"><?= e((string) $row['name']) ?></td>
                            <td style="text-align:right;">
                                <form method="post" onsubmit="return confirm('Bạn có chắc chắn muốn xóa thuốc này?');" style="display:inline;">
                                    <input type="hidden" name="action" value="delete">
                                    <input type="hidden" name="id" value="<?= e((string) $row['id']) ?>">
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