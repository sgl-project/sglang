import Foundation

enum AppLanguage: String, CaseIterable, Identifiable, Codable {
    case zhHans = "zh-Hans"
    case es = "es"
    case en = "en"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .zhHans: return "中文"
        case .es: return "Español"
        case .en: return "English"
        }
    }
}
