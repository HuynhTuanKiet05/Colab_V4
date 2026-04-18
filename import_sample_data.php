<?php

declare(strict_types=1);

require_once __DIR__ . '/app/bootstrap.php';

function importDrugs(PDO $pdo, string $file): void
{
    if (!file_exists($file)) {
        return;
    }
    $handle = fopen($file, 'r');
    fgetcsv($handle);
    $stmt = $pdo->prepare('INSERT INTO drugs (source_code, name, smiles) VALUES (:source_code, :name, :smiles) ON DUPLICATE KEY UPDATE name = VALUES(name), smiles = VALUES(smiles)');
    while (($row = fgetcsv($handle)) !== false) {
        $stmt->execute([
            'source_code' => $row[0] ?? '',
            'name' => $row[1] ?? '',
            'smiles' => $row[2] ?? null,
        ]);
    }
    fclose($handle);
}

function importDiseases(PDO $pdo, string $file): void
{
    if (!file_exists($file)) {
        return;
    }
    $handle = fopen($file, 'r');
    $stmt = $pdo->prepare('INSERT INTO diseases (source_code, name) VALUES (:source_code, :name) ON DUPLICATE KEY UPDATE name = VALUES(name)');
    while (($row = fgetcsv($handle)) !== false) {
        if (empty($row[0])) {
            continue;
        }
        $stmt->execute([
            'source_code' => $row[0],
            'name' => $row[0],
        ]);
    }
    fclose($handle);
}

function importProteins(PDO $pdo, string $file): void
{
    if (!file_exists($file)) {
        return;
    }
    $handle = fopen($file, 'r');
    fgetcsv($handle);
    $stmt = $pdo->prepare('INSERT INTO proteins (source_code, name, sequence_text) VALUES (:source_code, :name, :sequence_text) ON DUPLICATE KEY UPDATE sequence_text = VALUES(sequence_text)');
    while (($row = fgetcsv($handle)) !== false) {
        $stmt->execute([
            'source_code' => $row[0] ?? '',
            'name' => $row[0] ?? '',
            'sequence_text' => $row[1] ?? null,
        ]);
    }
    fclose($handle);
}

$datasetDir = __DIR__ . '/ductri_hgt_update/data/C-dataset';
$pdo = db();

importDrugs($pdo, $datasetDir . '/DrugInformation.csv');
importDiseases($pdo, $datasetDir . '/DiseaseFeature.csv');
importProteins($pdo, $datasetDir . '/ProteinInformation.csv');

echo "Imported sample data successfully.";
