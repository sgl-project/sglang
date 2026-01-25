import Foundation

struct LocalizedText: Codable, Hashable {
    let zh: String
    let es: String
    let en: String?

    func text(for language: AppLanguage) -> String {
        switch language {
        case .zhHans:
            return zh
        case .es:
            return es
        case .en:
            return en ?? zh
        }
    }
}
