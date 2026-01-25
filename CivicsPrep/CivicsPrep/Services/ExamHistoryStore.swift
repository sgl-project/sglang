import Foundation

final class ExamHistoryStore: ObservableObject {
    @Published private(set) var history: [ExamResult]

    init() {
        history = ExamHistoryStore.loadHistory()
    }

    func add(_ result: ExamResult) {
        var updated = history
        updated.insert(result, at: 0)
        history = Array(updated.prefix(10))
        ExamHistoryStore.saveHistory(history)
    }

    private static func loadHistory() -> [ExamResult] {
        guard let data = UserDefaults.standard.data(forKey: StorageKeys.examHistory) else {
            return []
        }
        return (try? JSONDecoder().decode([ExamResult].self, from: data)) ?? []
    }

    private static func saveHistory(_ results: [ExamResult]) {
        if let data = try? JSONEncoder().encode(results) {
            UserDefaults.standard.set(data, forKey: StorageKeys.examHistory)
        }
    }
}
