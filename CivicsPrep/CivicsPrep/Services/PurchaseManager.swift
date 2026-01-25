import Foundation
import StoreKit

@MainActor
final class PurchaseManager: ObservableObject {
    static let removeAdsID = "com.example.civicsprep.removeads"
    static let mockExamID = "com.example.civicsprep.mockexam"

    @Published private(set) var products: [Product] = []
    @Published private(set) var hasRemoveAds: Bool
    @Published private(set) var hasMockExam: Bool

    private var updateTask: Task<Void, Never>?

    init() {
        hasRemoveAds = UserDefaults.standard.bool(forKey: StorageKeys.removeAdsEntitlement)
        hasMockExam = UserDefaults.standard.bool(forKey: StorageKeys.mockExamEntitlement)
        updateTask = listenForTransactions()
        Task {
            await refreshEntitlements()
            await loadProducts()
        }
    }

    deinit {
        updateTask?.cancel()
    }

    func loadProducts() async {
        do {
            products = try await Product.products(for: [Self.removeAdsID, Self.mockExamID])
        } catch {
            products = []
        }
    }

    func purchase(_ product: Product) async -> Bool {
        do {
            let result = try await product.purchase()
            switch result {
            case .success(let verification):
                let transaction = try checkVerified(verification)
                await updateEntitlement(for: transaction.productID, value: true)
                await transaction.finish()
                return true
            case .userCancelled, .pending:
                return false
            @unknown default:
                return false
            }
        } catch {
            return false
        }
    }

    func restorePurchases() async {
        await refreshEntitlements()
    }

    func isProductPurchased(_ productID: String) -> Bool {
        switch productID {
        case Self.removeAdsID:
            return hasRemoveAds
        case Self.mockExamID:
            return hasMockExam
        default:
            return false
        }
    }

    private func refreshEntitlements() async {
        var removeAds = false
        var mockExam = false
        for await result in Transaction.currentEntitlements {
            if case .verified(let transaction) = result {
                if transaction.productID == Self.removeAdsID {
                    removeAds = true
                }
                if transaction.productID == Self.mockExamID {
                    mockExam = true
                }
            }
        }
        await updateEntitlement(for: Self.removeAdsID, value: removeAds)
        await updateEntitlement(for: Self.mockExamID, value: mockExam)
    }

    private func listenForTransactions() -> Task<Void, Never> {
        return Task.detached(priority: .background) { [weak self] in
            for await result in Transaction.updates {
                guard let self else { continue }
                if case .verified(let transaction) = result {
                    await self.updateEntitlement(for: transaction.productID, value: true)
                    await transaction.finish()
                }
            }
        }
    }

    private func updateEntitlement(for productID: String, value: Bool) async {
        switch productID {
        case Self.removeAdsID:
            hasRemoveAds = value
            UserDefaults.standard.set(value, forKey: StorageKeys.removeAdsEntitlement)
        case Self.mockExamID:
            hasMockExam = value
            UserDefaults.standard.set(value, forKey: StorageKeys.mockExamEntitlement)
        default:
            break
        }
    }

    private func checkVerified<T>(_ result: VerificationResult<T>) throws -> T {
        switch result {
        case .unverified:
            throw StoreKitError.userCancelled
        case .verified(let safe):
            return safe
        }
    }
}
