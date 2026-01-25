import Foundation

final class LocalizationManager: ObservableObject {
    @Published var language: AppLanguage {
        didSet {
            UserDefaults.standard.set(language.rawValue, forKey: StorageKeys.language)
        }
    }

    init() {
        if let saved = UserDefaults.standard.string(forKey: StorageKeys.language),
           let value = AppLanguage(rawValue: saved) {
            language = value
        } else {
            language = .zhHans
        }
    }

    func localized(_ key: String) -> String {
        let bundle = bundleForCurrentLanguage()
        return bundle.localizedString(forKey: key, value: nil, table: nil)
    }

    private func bundleForCurrentLanguage() -> Bundle {
        guard let path = Bundle.main.path(forResource: language.rawValue, ofType: "lproj"),
              let bundle = Bundle(path: path) else {
            return .main
        }
        return bundle
    }
}

enum StorageKeys {
    static let language = "app.language"
    static let bookmarks = "app.bookmarks"
    static let mastered = "app.mastered"
    static let removeAdsEntitlement = "app.entitlement.removeAds"
    static let mockExamEntitlement = "app.entitlement.mockExam"
    static let examHistory = "app.examHistory"
    static let onboardingSeen = "app.onboardingSeen"
}
