-- Drug-Disease Prediction WebApp Database Schema
-- Import file này vào phpMyAdmin trên XAMPP
-- Database đề xuất: drug_disease_ai

CREATE DATABASE IF NOT EXISTS `drug_disease_ai`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE `drug_disease_ai`;

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS `prediction_results`;
DROP TABLE IF EXISTS `prediction_requests`;
DROP TABLE IF EXISTS `drug_protein_links`;
DROP TABLE IF EXISTS `protein_disease_links`;
DROP TABLE IF EXISTS `drug_disease_links`;
DROP TABLE IF EXISTS `proteins`;
DROP TABLE IF EXISTS `diseases`;
DROP TABLE IF EXISTS `drugs`;
DROP TABLE IF EXISTS `users`;
DROP TABLE IF EXISTS `system_settings`;

CREATE TABLE `users` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `full_name` VARCHAR(150) NOT NULL,
  `username` VARCHAR(50) NOT NULL,
  `email` VARCHAR(120) DEFAULT NULL,
  `password_hash` VARCHAR(255) NOT NULL,
  `role` ENUM('admin', 'user') NOT NULL DEFAULT 'user',
  `status` ENUM('active', 'inactive') NOT NULL DEFAULT 'active',
  `last_login_at` DATETIME NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_users_username` (`username`),
  UNIQUE KEY `uk_users_email` (`email`)
) ENGINE=InnoDB;

CREATE TABLE `drugs` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `source_code` VARCHAR(50) NOT NULL,
  `name` VARCHAR(255) NOT NULL,
  `smiles` LONGTEXT NULL,
  `description` TEXT NULL,
  `is_active` TINYINT(1) NOT NULL DEFAULT 1,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_drugs_source_code` (`source_code`),
  KEY `idx_drugs_name` (`name`)
) ENGINE=InnoDB;

CREATE TABLE `diseases` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `source_code` VARCHAR(50) NOT NULL,
  `name` VARCHAR(255) NOT NULL,
  `description` TEXT NULL,
  `is_active` TINYINT(1) NOT NULL DEFAULT 1,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_diseases_source_code` (`source_code`),
  KEY `idx_diseases_name` (`name`)
) ENGINE=InnoDB;

CREATE TABLE `proteins` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `source_code` VARCHAR(50) NOT NULL,
  `name` VARCHAR(255) DEFAULT NULL,
  `sequence_text` LONGTEXT NULL,
  `description` TEXT NULL,
  `is_active` TINYINT(1) NOT NULL DEFAULT 1,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_proteins_source_code` (`source_code`),
  KEY `idx_proteins_name` (`name`)
) ENGINE=InnoDB;

CREATE TABLE `drug_disease_links` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `drug_id` BIGINT UNSIGNED NOT NULL,
  `disease_id` BIGINT UNSIGNED NOT NULL,
  `association_type` ENUM('known_positive', 'known_negative', 'predicted', 'validated') NOT NULL DEFAULT 'known_positive',
  `score` DECIMAL(10,6) DEFAULT NULL,
  `source_note` VARCHAR(255) DEFAULT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_drug_disease_pair` (`drug_id`, `disease_id`),
  KEY `idx_ddl_disease_id` (`disease_id`),
  KEY `idx_ddl_assoc_type` (`association_type`),
  CONSTRAINT `fk_ddl_drug` FOREIGN KEY (`drug_id`) REFERENCES `drugs` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_ddl_disease` FOREIGN KEY (`disease_id`) REFERENCES `diseases` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

CREATE TABLE `drug_protein_links` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `drug_id` BIGINT UNSIGNED NOT NULL,
  `protein_id` BIGINT UNSIGNED NOT NULL,
  `score` DECIMAL(10,6) DEFAULT NULL,
  `source_note` VARCHAR(255) DEFAULT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_drug_protein_pair` (`drug_id`, `protein_id`),
  KEY `idx_dpl_protein_id` (`protein_id`),
  CONSTRAINT `fk_dpl_drug` FOREIGN KEY (`drug_id`) REFERENCES `drugs` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_dpl_protein` FOREIGN KEY (`protein_id`) REFERENCES `proteins` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

CREATE TABLE `protein_disease_links` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `protein_id` BIGINT UNSIGNED NOT NULL,
  `disease_id` BIGINT UNSIGNED NOT NULL,
  `score` DECIMAL(10,6) DEFAULT NULL,
  `source_note` VARCHAR(255) DEFAULT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_protein_disease_pair` (`protein_id`, `disease_id`),
  KEY `idx_pdl_disease_id` (`disease_id`),
  CONSTRAINT `fk_pdl_protein` FOREIGN KEY (`protein_id`) REFERENCES `proteins` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_pdl_disease` FOREIGN KEY (`disease_id`) REFERENCES `diseases` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

CREATE TABLE `prediction_requests` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `user_id` BIGINT UNSIGNED NOT NULL,
  `query_type` ENUM('drug_to_disease', 'disease_to_drug') NOT NULL,
  `input_text` VARCHAR(255) NOT NULL,
  `matched_entity_id` BIGINT UNSIGNED DEFAULT NULL,
  `matched_entity_type` ENUM('drug', 'disease') DEFAULT NULL,
  `top_k` INT NOT NULL DEFAULT 10,
  `status` ENUM('pending', 'success', 'failed') NOT NULL DEFAULT 'success',
  `error_message` TEXT NULL,
  `request_ip` VARCHAR(45) DEFAULT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_prediction_requests_user_id` (`user_id`),
  KEY `idx_prediction_requests_query_type` (`query_type`),
  KEY `idx_prediction_requests_created_at` (`created_at`),
  CONSTRAINT `fk_prediction_requests_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

CREATE TABLE `prediction_results` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `request_id` BIGINT UNSIGNED NOT NULL,
  `rank_no` INT NOT NULL,
  `result_entity_type` ENUM('drug', 'disease') NOT NULL,
  `result_entity_id` BIGINT UNSIGNED DEFAULT NULL,
  `result_name` VARCHAR(255) NOT NULL,
  `score` DECIMAL(10,6) NOT NULL,
  `metadata_json` JSON DEFAULT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_prediction_result_rank` (`request_id`, `rank_no`),
  KEY `idx_prediction_results_entity_type` (`result_entity_type`),
  CONSTRAINT `fk_prediction_results_request` FOREIGN KEY (`request_id`) REFERENCES `prediction_requests` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

CREATE TABLE `system_settings` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `setting_key` VARCHAR(100) NOT NULL,
  `setting_value` TEXT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_system_settings_key` (`setting_key`)
) ENGINE=InnoDB;

INSERT INTO `users` (`full_name`, `username`, `email`, `password_hash`, `role`, `status`)
VALUES
('Administrator', 'admin', 'admin@example.com', '$2y$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', 'admin', 'active'),
('Người dùng mẫu', 'user1', 'user1@example.com', '$2y$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', 'user', 'active');

INSERT INTO `system_settings` (`setting_key`, `setting_value`) VALUES
('app_name', 'Drug Disease AI Predictor'),
('default_dataset', 'C-dataset'),
('default_top_k', '10'),
('python_api_base_url', 'http://127.0.0.1:8000');

SET FOREIGN_KEY_CHECKS = 1;

-- Tài khoản mặc định:
-- admin / password
-- user1 / password
