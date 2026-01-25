import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var localization: LocalizationManager
    @EnvironmentObject private var purchaseManager: PurchaseManager
    @EnvironmentObject private var settingsViewModel: SettingsViewModel
    @EnvironmentObject private var examHistory: ExamHistoryStore

    var body: some View {
        NavigationStack {
            List {
                Section(localization.localized("language")) {
                    Picker(localization.localized("language"), selection: $localization.language) {
                        Text(localization.localized("language_zh")).tag(AppLanguage.zhHans)
                        Text(localization.localized("language_es")).tag(AppLanguage.es)
                    }
                }
                Section(localization.localized("upgrades")) {
                    ForEach(purchaseManager.products, id: \.id) { product in
                        HStack {
                            VStack(alignment: .leading) {
                                Text(title(for: product.id))
                                Text(product.displayPrice)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            if purchaseManager.isProductPurchased(product.id) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(.green)
                            } else {
                                Button(localization.localized("purchase_unlock")) {
                                    Task { await settingsViewModel.purchase(product: product) }
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                    }
                    Button(localization.localized("restore_purchases")) {
                        Task { await settingsViewModel.restore() }
                    }
                }
                Section(localization.localized("history")) {
                    if examHistory.history.isEmpty {
                        Text(localization.localized("no_history"))
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(examHistory.history) { result in
                            VStack(alignment: .leading) {
                                Text(result.date, style: .date)
                                Text("\(localization.localized("score")): \(result.correctCount)/\(result.totalQuestions)")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                Section {
                    Link(localization.localized("privacy"), destination: URL(string: "https://example.com/privacy")!)
                    Link(localization.localized("terms"), destination: URL(string: "https://example.com/terms")!)
                }
            }
            .navigationTitle(localization.localized("settings_title"))
            .alert(localization.localized(settingsViewModel.purchaseMessage), isPresented: $settingsViewModel.showPurchaseResult) {
                Button("OK", role: .cancel) { }
            }
        }
    }

    private func title(for productID: String) -> String {
        switch productID {
        case PurchaseManager.removeAdsID:
            return localization.localized("remove_ads")
        case PurchaseManager.mockExamID:
            return localization.localized("mock_exam")
        default:
            return productID
        }
    }
}
