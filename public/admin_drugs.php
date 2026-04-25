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
        flash('success', 'Đã xoá thuốc.');
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
    <title>Quản lý thuốc · AMNTDDA AI</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">Quản lý thuốc</div>
            <div class="muted">Thêm, xoá và theo dõi các thực thể thuốc trong cơ sở dữ liệu.</div>
        </div>
        <div class="nav-links">
            <a class="btn btn-ghost" href="admin.php">Quay lại</a>
            <a class="btn btn-danger" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <?php if ($success): ?><div class="alert alert-success"><?= e($success) ?></div><?php endif; ?>

    <div class="grid split-panel">
        <div class="glass-card">
            <h3>Thêm thực thể thuốc mới</h3>
            <p class="muted spacer-lg">Điền thông tin định danh và đặc trưng hoá học cơ bản để bổ sung vào dữ liệu quản trị.</p>
            <form method="post">
                <input type="hidden" name="action" value="create">
                <div class="stack-tight">
                    <div class="form-group">
                        <label class="label">Mã nguồn</label>
                        <input class="input" name="source_code" placeholder="Ví dụ: DB00014" required>
                    </div>
                    <div class="form-group">
                        <label class="label">Tên thuốc</label>
                        <input class="input" name="name" placeholder="Ví dụ: Goserelin" required>
                    </div>
                    <div class="form-group">
                        <label class="label">Cấu trúc SMILES</label>
                        <textarea class="input" name="smiles" placeholder="Dành cho thông tin cấu trúc phân tử..."></textarea>
                    </div>
                    <div class="form-group">
                        <label class="label">Mô tả</label>
                        <textarea class="input" name="description" placeholder="Thông tin bổ sung về thuốc..."></textarea>
                    </div>
                    <button class="btn btn-full" type="submit">Lưu thông tin</button>
                </div>
            </form>
        </div>

        <div class="glass-card">
            <div class="section-header">
                <div>
                    <h3>Danh sách thuốc</h3>
                    <p class="muted">Hiển thị tối đa 50 bản ghi mới nhất.</p>
                </div>
                <div class="badge badge-drug"><?= count($rows) ?> bản ghi</div>
            </div>

            <div class="table-container table-scroll">
                <table class="table">
                    <thead>
                    <tr>
                        <th>ID</th>
                        <th>Mã nguồn</th>
                        <th>Tên thuốc</th>
                        <th class="align-right">Thao tác</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php if (empty($rows)): ?>
                        <tr><td colspan="4" class="center-empty">Chưa có dữ liệu thuốc.</td></tr>
                    <?php endif; ?>
                    <?php foreach ($rows as $row): ?>
                        <tr>
                            <td class="muted">#<?= $row['id'] ?></td>
                            <td><span class="badge badge-drug"><?= e((string) $row['source_code']) ?></span></td>
                            <td><strong><?= e((string) $row['name']) ?></strong></td>
                            <td class="align-right">
                                <form method="post" class="inline-form" onsubmit="return confirm('Bạn có chắc chắn muốn xoá thuốc này?');">
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
