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
                flash('success', 'Da tao lien ket moi thanh cong.');
            } catch (PDOException $e) {
                if ($e->getCode() == 23000) {
                    flash('error', 'Lien ket giua hai thuc the nay da ton tai.');
                } else {
                    flash('error', 'Loi CSDL: ' . $e->getMessage());
                }
            }
        } else {
            flash('error', 'Vui long chon day du Thuoc va Benh.');
        }
    }

    if ($action === 'delete') {
        $id = (int) ($_POST['id'] ?? 0);
        $stmt = db()->prepare('DELETE FROM drug_disease_links WHERE id = :id');
        $stmt->execute(['id' => $id]);
        flash('success', 'Da xoa lien ket.');
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
    <title>Admin - Quan ly lien ket sinh hoc</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">Quan ly lien ket sinh hoc</div>
            <div class="muted">Tao, kiem soat va lam sach cac lien ket Thuoc - Benh trong co so du lieu.</div>
        </div>
        <div class="nav-links">
            <a class="btn btn-ghost" href="admin.php">Quay lai</a>
            <a class="btn btn-danger" href="logout.php">Dang xuat</a>
        </div>
    </div>

    <?php if ($success): ?><div class="alert alert-success"><?= e($success) ?></div><?php endif; ?>
    <?php if ($error): ?><div class="alert alert-error"><?= e($error) ?></div><?php endif; ?>

    <div class="grid split-panel">
        <div class="glass-card">
            <h3>Tao lien ket Thuoc - Benh</h3>
            <p class="muted spacer-lg">Xac lap moi quan he de bo sung du lieu huan luyen, doi chieu ground truth va phuc vu kiem chung.</p>
            <form method="post">
                <input type="hidden" name="action" value="create">
                <div class="stack-tight">
                    <div class="form-group">
                        <label class="label">Chon Thuoc</label>
                        <select class="select" name="drug_id" required>
                            <option value="">-- Chon thuoc --</option>
                            <?php foreach ($drugs as $d): ?>
                                <option value="<?= $d['id'] ?>"><?= e($d['name']) ?> (<?= e($d['source_code']) ?>)</option>
                            <?php endforeach; ?>
                        </select>
                    </div>

                    <div class="form-group">
                        <label class="label">Chon Benh ly</label>
                        <select class="select" name="disease_id" required>
                            <option value="">-- Chon benh --</option>
                            <?php foreach ($diseases as $d): ?>
                                <option value="<?= $d['id'] ?>"><?= e($d['name']) ?> (<?= e($d['source_code']) ?>)</option>
                            <?php endforeach; ?>
                        </select>
                    </div>

                    <div class="form-group">
                        <label class="label">Loai lien ket</label>
                        <select class="select" name="association_type">
                            <option value="known_positive">Da biet duong tinh</option>
                            <option value="known_negative">Da biet am tinh</option>
                            <option value="predicted">Do mo hinh du doan</option>
                            <option value="validated">Da duoc kiem chung</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label class="label">Do tin cay</label>
                        <input class="input" type="number" step="0.0001" name="score" value="1.0000">
                    </div>

                    <div class="form-group">
                        <label class="label">Ghi chu</label>
                        <input class="input" name="source_note" placeholder="Vi du: PubMed ID, ClinicalTrials.gov...">
                    </div>

                    <button class="btn btn-full" type="submit">Xac lap lien ket</button>
                </div>
            </form>
        </div>

        <div class="glass-card">
            <div class="section-header">
                <div>
                    <h3>Danh sach lien ket thuc te</h3>
                    <p class="muted">Hien thi cac lien ket moi nhat trong co so du lieu.</p>
                </div>
                <div class="badge badge-drug"><?= count($links) ?> records</div>
            </div>

            <div class="table-container table-scroll">
                <table class="table">
                    <thead>
                    <tr>
                        <th>Thuoc</th>
                        <th>Benh ly</th>
                        <th>Loai</th>
                        <th>Score</th>
                        <th class="align-right">Hanh dong</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php if (empty($links)): ?>
                        <tr><td colspan="5" class="center-empty">Chua co lien ket nao duoc xac lap.</td></tr>
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
                                <form method="post" class="inline-form" onsubmit="return confirm('Xoa lien ket nay?');">
                                    <input type="hidden" name="action" value="delete">
                                    <input type="hidden" name="id" value="<?= e((string) $row['id']) ?>">
                                    <button class="btn btn-danger btn-sm" type="submit">Xoa</button>
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
