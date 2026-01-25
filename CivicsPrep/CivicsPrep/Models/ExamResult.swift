import Foundation

struct ExamResult: Identifiable, Codable, Hashable {
    let id: UUID
    let date: Date
    let totalQuestions: Int
    let correctCount: Int
    let incorrectQuestionIDs: [String]

    var scorePercent: Int {
        guard totalQuestions > 0 else { return 0 }
        return Int((Double(correctCount) / Double(totalQuestions)) * 100.0)
    }
}
