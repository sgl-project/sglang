import Foundation
import Combine

final class StudyViewModel: ObservableObject {
    @Published private(set) var questions: [Question] = []
    @Published var searchText: String = ""
    @Published var selectedCategory: String? = nil

    private let repository: QuestionRepositoryProtocol

    init(repository: QuestionRepositoryProtocol) {
        self.repository = repository
        load()
    }

    func load() {
        do {
            questions = try repository.loadQuestions()
        } catch {
            questions = []
        }
    }

    var categories: [Category] {
        let titles = Set(questions.map { $0.category })
        return titles.sorted().map { Category(id: $0, title: $0) }
    }

    func filteredQuestions(language: AppLanguage) -> [Question] {
        let base = questions.filter { question in
            if let selectedCategory {
                return question.category == selectedCategory
            }
            return true
        }
        guard !searchText.isEmpty else { return base }
        return base.filter { question in
            let text = question.question.text(for: language)
            return text.localizedCaseInsensitiveContains(searchText)
        }
    }
}
