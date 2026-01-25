import Foundation

struct Question: Identifiable, Codable, Hashable {
    let id: String
    let category: String
    let difficulty: Int
    let question: LocalizedText
    let answers: [LocalizedText]
    let explanation: LocalizedText?
}

struct Category: Identifiable, Hashable {
    let id: String
    let title: String
}
