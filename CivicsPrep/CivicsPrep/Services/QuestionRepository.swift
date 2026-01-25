import Foundation

protocol QuestionRepositoryProtocol {
    func loadQuestions() throws -> [Question]
}

final class QuestionRepository: QuestionRepositoryProtocol {
    private let resourceName: String

    init(resourceName: String = "questions") {
        self.resourceName = resourceName
    }

    func loadQuestions() throws -> [Question] {
        guard let url = Bundle.main.url(forResource: resourceName, withExtension: "json") else {
            return []
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([Question].self, from: data)
    }
}
