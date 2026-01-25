import Foundation
import StoreKit

@MainActor
final class SettingsViewModel: ObservableObject {
    @Published var showPurchaseResult: Bool = false
    @Published var purchaseMessage: String = ""

    let purchaseManager: PurchaseManager

    init(purchaseManager: PurchaseManager) {
        self.purchaseManager = purchaseManager
    }

    func purchase(product: Product) async {
        let success = await purchaseManager.purchase(product)
        purchaseMessage = success ? "purchase_success" : "purchase_failed"
        showPurchaseResult = true
    }

    func restore() async {
        await purchaseManager.restorePurchases()
    }
}
